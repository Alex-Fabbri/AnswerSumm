# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import numpy as np
import torch

from fairseq.data import data_utils, FairseqDataset


logger = logging.getLogger(__name__)


def collate(
    samples,
    pad_idx,
    eos_idx,
    left_pad_source=True,
    left_pad_target=False,
    input_feeding=True, sentence_prediction=False, 
    sentence_prediction_pool_target_eos=False, 
    sentence_prediction_pool_source_eos=False,
    use_encoder_features=False, sentence_prediction_binary=False, 
    sentence_regression=False, every_timestep=False,
):
    if len(samples) == 0:
        return {}
    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    def check_alignment(alignment, src_len, tgt_len):
        if alignment is None or len(alignment) == 0:
            return False
        if alignment[:, 0].max().item() >= src_len - 1 or alignment[:, 1].max().item() >= tgt_len - 1:
            logger.warning("alignment size mismatch found, skipping alignment!")
            return False
        return True

    def compute_alignment_weights(alignments):
        """
        Given a tensor of shape [:, 2] containing the source-target indices
        corresponding to the alignments, a weight vector containing the
        inverse frequency of each target index is computed.
        For e.g. if alignments = [[5, 7], [2, 3], [1, 3], [4, 2]], then
        a tensor containing [1., 0.5, 0.5, 1] should be returned (since target
        index 3 is repeated twice)
        """
        align_tgt = alignments[:, 1]
        _, align_tgt_i, align_tgt_c = torch.unique(align_tgt, return_inverse=True, return_counts=True)
        align_weights = align_tgt_c[align_tgt_i[np.arange(len(align_tgt))]]
        return 1. / align_weights.float()

    id = torch.LongTensor([s['id'] for s in samples])
    # TODO if debugging
    #cur_ids = [s['id'] for s in samples] 
    #if 2763 in cur_ids:
    #    from fairseq import pdb;pdb.set_trace()
    # if cur_ids == [23994, 16914, 54802]: 
    # if sentence_regression and cur_ids == [77980]:
    span_regression_final = None
    prev_output_tokens = None
    target = None
    target_sent_ids_out = None
    span_start_out = None
    span_end_out = None
    dummy_targets = None
    prev_input_tokens = None
    summarization_ids_out = None
    
    src_tokens = merge('source', left_pad=left_pad_source)
    
    src_lengths_tmp = torch.LongTensor([
        s['source'].ne(pad_idx).long().sum() for s in samples
    ])
    # sort by descending source length
    src_lengths, sort_order = src_lengths_tmp.sort(descending=True)
    id = id.index_select(0, sort_order)

    src_tokens = src_tokens.index_select(0, sort_order)

    if samples[0].get('target', None) is not None:
        target = merge('target', left_pad=left_pad_target)
        target = target.index_select(0, sort_order)
        tgt_lengths = torch.LongTensor([
            s['target'].ne(pad_idx).long().sum() for s in samples
        ]).index_select(0, sort_order)
        ntokens = tgt_lengths.sum().item()

        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                'target',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
    else:
        ntokens = src_lengths.sum().item()
    
    # !! Span-related changes start here !!
    # Get previous input tokens for span prediction over source, analogous to prev_output_tokens
    if not use_encoder_features:
        prev_input_tokens = merge(
            'source',
            left_pad=left_pad_target,
            move_eos_to_beginning=True,
        )
        prev_input_tokens = prev_input_tokens.index_select(0, sort_order)


    # Get gold start and end positions for calculating loss, None at inference time
    if samples[0].get('span_item', None) is not None and not sentence_regression:
        span_tmp = [s['span_item'] for s in samples]
        start_spans = []
        end_spans = []
        # src_lengths_tmp because we add it to spans before sort_order
        offsets_unordered = src_lengths_tmp.max() - src_lengths_tmp
        for span, offset in zip(span_tmp, offsets_unordered):
            # span is just a list of [start0, end0, start1, end1, ...]
            if use_encoder_features:
                cur_span_start = [span[i]+1+offset for i in range(0, len(span)-1, 2)]
                cur_span_end = [span[i+1]+1+offset for i in range(0, len(span)-1, 2)]
            else:
                # here we don't need the offset because we are using spans on the decoder side (left_pad_target is False)
                # but we do need an extra +1 for </s>
                cur_span_start = [span[i]+2 for i in range(0, len(span)-1, 2)]
                cur_span_end = [span[i+1]+2 for i in range(0, len(span)-1, 2)]
            for count, (cur_start, cur_end) in enumerate(zip(cur_span_start, cur_span_end)):
                # I'm going to just remove both since maybe the span calculations were getting mixed up because of this
                if cur_end >= src_lengths.max() or cur_start >= src_lengths.max(): # This is because the indices from preprocessing are before adding bos
                    cur_span_end[count] = torch.tensor(2000)
                    cur_span_start[count] = torch.tensor(2000)
            cur_span_start_stacked = torch.stack(cur_span_start, dim=0).T
            cur_span_end_stacked = torch.stack(cur_span_end, dim=0).T
            start_spans.append(cur_span_start_stacked)
            end_spans.append(cur_span_end_stacked)
        # TODO also change the pad_idx to 2000
        span_start_out = data_utils.collate_tokens(
                start_spans,
                2000, eos_idx, False, False,
            )
        span_end_out = data_utils.collate_tokens(
                end_spans,
                2000, eos_idx, False, False,
            )
        span_start_out = span_start_out.index_select(0, sort_order)
        span_end_out = span_end_out.index_select(0, sort_order)    
    
    # Get target ids corresponding to <S> from the target (using prev_output_tokens as this is what is fed to decoder)
    # just being very paranoid about cloning due to errors later on, but this isn't necessary
    #from fairseq import pdb;pdb.set_trace()
    if samples[0].get('target', None) is not None:
        target_sent_ids = []
        cur_output_tokens = prev_output_tokens.clone()
        for i in range(cur_output_tokens.shape[0]):
            # hard code the id for "<S>"
            tmp_tensor = (cur_output_tokens[i]==50257).nonzero().squeeze()
            if len(tmp_tensor.shape) == 0:
                tmp_tensor = tmp_tensor.unsqueeze(-1)
            if sentence_prediction_pool_target_eos:
                # add representation for starting target (skip bos)
                start_tok = torch.LongTensor([1])  
                tmp_tensor = torch.cat([start_tok, tmp_tensor], dim=0)
            target_sent_ids.append(tmp_tensor)
        target_sent_ids_out = data_utils.collate_tokens(
                target_sent_ids,
                -43, eos_idx, False, False,
            )

    if sentence_prediction and not sentence_regression:
        if sentence_prediction_binary:
            if use_encoder_features:
                cur_tokens = src_tokens
            else:
                cur_tokens = prev_input_tokens
            if samples[0].get('span_item', None) is not None:
                summarization_targets =  torch.zeros(span_end_out.shape[0], span_end_out.shape[1], max(1, max([sum(x) for x in (src_tokens==50257)])/2), dtype=torch.int64)
                summarization_targets = summarization_targets + 2000
            summarization_ids = []
            for i in range(cur_tokens.shape[0]):
                # Get ids so we can extract EOS from the source
                src_token_seps = (cur_tokens[i]==50257).nonzero().squeeze()
                if src_token_seps.shape[0] == 0:
                    src_token_seps = torch.tensor([1, 2], dtype=torch.int64)
                    src_token_seps_single = [1]
                else:
                    if src_token_seps.shape[0] % 2 != 0:
                        src_token_seps = src_token_seps[:-1]
                    # Get EOS as opposed to BOS
                    src_token_seps_single = [src_token_seps[i+1] for i in range(0, len(src_token_seps)-1, 2)]

                if sentence_prediction_pool_source_eos:
                    # if we are pooling we keep both the BOS and EOS (the symbols are the same but one is at the start, one at the end)
                    summarization_ids.append(src_token_seps)
                else:
                    if len(src_token_seps_single) == 1:
                        summarization_ids.append(torch.tensor(src_token_seps_single))
                    else:    
                        summarization_ids.append(torch.stack(src_token_seps_single))
                
                # Insert positive labels 
                if samples[0].get('span_item', None) is not None:
                    for span_count, cur_span in enumerate(span_end_out[i]):
                        if cur_span == torch.tensor([2000]):
                            continue
                        summarization_targets[i, span_count, :len(src_token_seps_single)] = 0
                        try:
                            summarization_targets[i, span_count, src_token_seps_single.index(cur_span)] = 1
                        except:
                            # sometimes the src span gets cut off, so then the whole span example is ignored
                            summarization_targets[i, span_count, :len(src_token_seps_single)] = 2000
            summarization_ids_out = data_utils.collate_tokens(
                        summarization_ids,
                        -43, eos_idx, False, False,
                    )
        else:
            if use_encoder_features:
                cur_tokens = src_tokens
            else:
                cur_tokens = prev_input_tokens
            summarization_targets = []
            summarization_ids = []
            for i in range(cur_tokens.shape[0]):
                # Get ids so we can extract EOS from the source
                src_token_seps = (cur_tokens[i]==50257).nonzero().squeeze()
                if src_token_seps.shape[0] == 0:
                    src_token_seps = torch.tensor([1, 2], dtype=torch.int64)
                    src_token_seps_single = [1]
                else:
                    if src_token_seps.shape[0] % 2 != 0:
                        src_token_seps = src_token_seps[:-1]
                    # Get EOS as opposed to BOS
                    src_token_seps_single = [src_token_seps[i+1] for i in range(0, len(src_token_seps)-1, 2)]

                if sentence_prediction_pool_source_eos:
                    # if we are pooling we keep both the BOS and EOS (the symbols are the same but one is at the start, one at the end)
                    summarization_ids.append(src_token_seps)
                else:
                    if len(src_token_seps_single) == 1:
                        summarization_ids.append(torch.tensor(src_token_seps_single))
                    else:    
                        summarization_ids.append(torch.stack(src_token_seps_single))
                # Insert positive labels 
                cur_labels = []
                if samples[0].get('span_item', None) is not None:
                    for span_count, cur_span in enumerate(span_end_out[i]):
                        if cur_span == torch.tensor([2000]):
                            cur_labels.append(2000)
                            continue
                        try:
                            cur_labels.append(src_token_seps_single.index(cur_span))
                        except:
                            cur_labels.append(2000)
                summarization_targets.append(torch.tensor(cur_labels, dtype=torch.int64))
            summarization_ids_out = data_utils.collate_tokens(
                        summarization_ids,
                        -43, eos_idx, False, False,
                    )
            # because it's an index I put it as 2000, but could be a negative number too
            summarization_targets = data_utils.collate_tokens(
                    summarization_targets,
                    2000, eos_idx, False, False,
                    )
    if sentence_regression:
        if use_encoder_features:
            cur_tokens = src_tokens
        else: 
            cur_tokens = prev_input_tokens
        summarization_ids = []
        for i in range(cur_tokens.shape[0]):
            # Get ids so we can extract EOS from the source
            src_token_seps = (cur_tokens[i]==50257).nonzero().squeeze()
            if src_token_seps.shape[0] == 0:
                src_token_seps = torch.tensor([1, 2], dtype=torch.int64)
                src_token_seps_single = [1]
            else:
                if src_token_seps.shape[0] % 2 != 0:
                    src_token_seps = src_token_seps[:-1]
                # Get EOS as opposed to BOS
                src_token_seps_single = [src_token_seps[i+1] for i in range(0, len(src_token_seps)-1, 2)]
            if sentence_prediction_pool_source_eos:
                # if we are pooling we keep both the BOS and EOS (the symbols are the same but one is at the start, one at the end)
                summarization_ids.append(src_token_seps)
            else:
                if len(src_token_seps_single) == 1:
                    summarization_ids.append(torch.tensor(src_token_seps_single))
                else:    
                    summarization_ids.append(torch.stack(src_token_seps_single))
        summarization_ids_out = data_utils.collate_tokens(
                    summarization_ids,
                    -43, eos_idx, False, False,
                )

        span_regression = []
        span_tmp = [s['span_item'] for s in samples]
        if span_tmp[0] is not None:
            sort_order_list = sort_order.tolist()
            span_tmp_sorted = [span_tmp[x] for x in sort_order_list]
            max_sents = summarization_ids_out.shape[-1]
            if every_timestep:
                max_spans = target.shape[-1]
            else:
                max_spans = target_sent_ids_out.shape[-1]
            for span_item in span_tmp_sorted:
                span_item = span_item.tolist()
                result = []
                tmp = []
                for entry in span_item:
                    if entry != -1000:
                        tmp.append(entry)
                    else:
                        result.append(tmp)
                        tmp = []
                result.append(tmp)
                lens = [len(x) for x in result]
                assert sum(lens) == (len(result[0]) * len(result))
                span_regression.append(result)
            span_regression_final = torch.zeros(len(span_tmp), max_spans, max_sents)
        
            # iterate over batch
            if every_timestep:
                for i in range(target_sent_ids_out.shape[0]):
                    cur_span_regression = span_regression[i]
                    cur_target_sent_ids_out = target_sent_ids_out[i]
                    starter = 0
                    for span_reg, stopper in zip(cur_span_regression, cur_target_sent_ids_out):
                        cur_t = torch.tensor(span_reg).unsqueeze(0)
                        cur_t = cur_t.repeat(stopper+1-starter, 1)
                        try:
                            span_regression_final[i, starter:stopper+1, :min(cur_t.shape[-1], max_sents)] = cur_t[:, :max_sents]
                        except:
                            from fairseq import pdb;pdb.set_trace()
                        starter = stopper + 1
            else:
                for count, item in enumerate(span_regression):
                    item = np.array(item)
                    span_regression_final[count, :min(len(item), max_spans) , :min(len(item[0]), max_sents)] = torch.tensor(item[:max_spans, :max_sents])
        
        # span_regression = []
        # span_tmp = [s['span_item'] for s in samples]
        # max_sents = summarization_ids_out.shape[-1]
        # max_spans = target_sent_ids_out.shape[-1]
        # for span_item in span_tmp:
        #     span_item = span_item.tolist()
        #     result = []
        #     tmp = []
        #     for entry in span_item:
        #         if entry != -1000:
        #             tmp.append(entry)
        #         else:
        #             result.append(tmp)
        #             tmp = []
        #     result.append(tmp)
        #     lens = [len(x) for x in result]
        #     assert sum(lens) == (len(result[0]) * len(result))
        #     span_regression.append(result)
        # span_regression_final = torch.zeros(len(span_tmp), max_spans, max_sents)
        # for count, item in enumerate(span_regression):
        #     item = np.array(item)
        #     span_regression_final[count, :min(len(item), max_spans) , :min(len(item[0]), max_sents)] = torch.tensor(item[:max_spans, :max_sents])
        # span_regression_final = span_regression_final.index_select(0, sort_order)

    # trying a dummy classifier
    # if samples[0].get('target', None) is not None:
    #     # if condition bc otherwise cur_output_tokens is None
    #     dummy_targets = []
    #     for i in range(cur_output_tokens.shape[0]):
    #         final_tok = torch.LongTensor([0])  
    #         dummy_targets.append(final_tok)
    #     dummy_targets = torch.stack(dummy_targets, dim=0)
    #     dummy_targets = dummy_targets.squeeze(-1)
    #     dummy_targets = dummy_targets.index_select(0, sort_order)
    if every_timestep and not sentence_regression:
        span_out_start_all = torch.zeros_like(target)
        span_out_end_all = torch.zeros_like(target)
        for i in range(target_sent_ids_out.shape[0]):
            cur_target_sent_ids_out = target_sent_ids_out[i]
            cur_span_start = span_start_out[i]
            cur_span_end = span_end_out[i]
            starter = 0
            for span_, stopper in zip(cur_span_start, cur_target_sent_ids_out):
                span_out_start_all[i, starter:stopper+1] = span_
                starter = stopper + 1
            starter = 0
            for span_, stopper in zip(cur_span_end, cur_target_sent_ids_out):
                span_out_end_all[i, starter:stopper+1] = span_
                starter = stopper + 1
    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
        },
        'target': target,
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens
    if prev_input_tokens is not None:
        batch['net_input']['prev_input_tokens'] = prev_input_tokens
    if target_sent_ids_out is not None:
        batch['net_input']['target_sent_ids'] = target_sent_ids_out
    if sentence_prediction:
        batch['net_input']['summarization_ids_out'] = summarization_ids_out
        if samples[0].get('span_item', None) is not None:
            if not sentence_regression:
                batch['summarization_targets'] = summarization_targets
    if span_start_out is not None:
        if every_timestep:
            batch['span_start_ids'] = span_out_start_all
        else:
            batch['span_start_ids'] = span_start_out
    if span_end_out is not None:
        if every_timestep:
            batch['span_end_ids'] = span_out_end_all
        else:
            batch['span_end_ids'] = span_end_out
    if span_regression_final is not None:
        batch['span_regression_final'] = span_regression_final
    if dummy_targets is not None:
        batch['dummy_targets'] = dummy_targets

    if samples[0].get('alignment', None) is not None:
        bsz, tgt_sz = batch['target'].shape
        src_sz = batch['net_input']['src_tokens'].shape[1]

        offsets = torch.zeros((len(sort_order), 2), dtype=torch.long)
        offsets[:, 1] += (torch.arange(len(sort_order), dtype=torch.long) * tgt_sz)
        if left_pad_source:
            offsets[:, 0] += (src_sz - src_lengths)
        if left_pad_target:
            offsets[:, 1] += (tgt_sz - tgt_lengths)

        alignments = [
            alignment + offset
            for align_idx, offset, src_len, tgt_len in zip(sort_order, offsets, src_lengths, tgt_lengths)
            for alignment in [samples[align_idx]['alignment'].view(-1, 2)]
            if check_alignment(alignment, src_len, tgt_len)
        ]

        if len(alignments) > 0:
            alignments = torch.cat(alignments, dim=0)
            align_weights = compute_alignment_weights(alignments)

            batch['alignments'] = alignments
            batch['align_weights'] = align_weights

    return batch


class LanguagePairSpanDataset(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets.

    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        src_dict (~fairseq.data.Dictionary): source vocabulary
        tgt (torch.utils.data.Dataset, optional): target dataset to wrap
        tgt_sizes (List[int], optional): target sentence lengths
        tgt_dict (~fairseq.data.Dictionary, optional): target vocabulary
        span (torch.utils.data.Dataset): span dataset to wrap
        left_pad_source (bool, optional): pad source tensors on the left side
            (default: True).
        left_pad_target (bool, optional): pad target tensors on the left side
            (default: False).
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for teacher forcing (default: True).
        remove_eos_from_source (bool, optional): if set, removes eos from end
            of source if it's present (default: False).
        append_eos_to_target (bool, optional): if set, appends eos to end of
            target if it's absent (default: False).
        align_dataset (torch.utils.data.Dataset, optional): dataset
            containing alignments.
        append_bos (bool, optional): if set, appends bos to the beginning of
            source/target sentence.
        num_buckets (int, optional): if set to a value greater than 0, then
            batches will be bucketed into the given number of batch shapes.
    """

    def __init__(
        self, src, src_sizes, src_dict,
        tgt=None, tgt_sizes=None, tgt_dict=None, span=None,
        left_pad_source=True, left_pad_target=False,
        shuffle=True, input_feeding=True,
        remove_eos_from_source=False, append_eos_to_target=False,
        align_dataset=None,
        append_bos=False, eos=None,
        num_buckets=0, sentence_prediction=False,
        sentence_prediction_pool_target_eos=False, 
        sentence_prediction_pool_source_eos=False,
        use_encoder_features=False, sentence_prediction_binary=False, 
        sentence_regression=False, every_timestep=False,
    ):
        
        if tgt_dict is not None:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()
        if tgt is not None:
            assert len(src) == len(tgt), "Source and target must contain the same number of examples"
        self.src = src
        self.tgt = tgt
        self.span = span
        self.sentence_prediction = sentence_prediction
        self.sentence_prediction_pool_target_eos = sentence_prediction_pool_target_eos
        self.sentence_prediction_pool_source_eos = sentence_prediction_pool_source_eos
        self.sentence_prediction_binary = sentence_prediction_binary
        self.sentence_regression = sentence_regression
        self.use_encoder_features = use_encoder_features
        self.every_timestep = every_timestep
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source
        self.append_eos_to_target = append_eos_to_target
        self.align_dataset = align_dataset
        if self.align_dataset is not None:
            assert self.tgt_sizes is not None, "Both source and target needed when alignments are provided"
        self.append_bos = append_bos
        self.eos = (eos if eos is not None else src_dict.eos())

        if num_buckets > 0:
            from fairseq.data import BucketPadLengthDataset
            self.src = BucketPadLengthDataset(
                self.src,
                sizes=self.src_sizes,
                num_buckets=num_buckets,
                pad_idx=self.src_dict.pad(),
                left_pad=self.left_pad_source,
            )
            self.src_sizes = self.src.sizes
            logger.info('bucketing source lengths: {}'.format(list(self.src.buckets)))
            if self.tgt is not None:
                self.tgt = BucketPadLengthDataset(
                    self.tgt,
                    sizes=self.tgt_sizes,
                    num_buckets=num_buckets,
                    pad_idx=self.tgt_dict.pad(),
                    left_pad=self.left_pad_target,
                )
                self.tgt_sizes = self.tgt.sizes
                logger.info('bucketing target lengths: {}'.format(list(self.tgt.buckets)))

            # determine bucket sizes using self.num_tokens, which will return
            # the padded lengths (thanks to BucketPadLengthDataset)
            num_tokens = np.vectorize(self.num_tokens, otypes=[np.long])
            self.bucketed_num_tokens = num_tokens(np.arange(len(self.src)))
            self.buckets = [
                (None, num_tokens)
                for num_tokens in np.unique(self.bucketed_num_tokens)
            ]
        else:
            self.buckets = None

    def get_batch_shapes(self):
        return self.buckets

    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        src_item = self.src[index]
        span_item = self.span[index] if self.span is not None else None
        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        if self.append_eos_to_target:
            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
            if self.tgt and self.tgt[index][-1] != eos:
                tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])

        if self.append_bos:
            bos = self.tgt_dict.bos() if self.tgt_dict else self.src_dict.bos()
            if self.tgt and self.tgt[index][0] != bos:
                tgt_item = torch.cat([torch.LongTensor([bos]), self.tgt[index]])

            bos = self.src_dict.bos()
            if self.src[index][-1] != bos:
                src_item = torch.cat([torch.LongTensor([bos]), self.src[index]])

        if self.remove_eos_from_source:
            eos = self.src_dict.eos()
            if self.src[index][-1] == eos:
                src_item = self.src[index][:-1]

        example = {
            'id': index,
            'source': src_item,
            'target': tgt_item,
            'span_item': span_item
        }
        if self.align_dataset is not None:
            example['alignment'] = self.align_dataset[index]
        return example

    def __len__(self):
        return len(self.src)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one
                    position for teacher forcing, of shape `(bsz, tgt_len)`.
                    This key will not be present if *input_feeding* is
                    ``False``.  Padding will appear on the left if
                    *left_pad_target* is ``True``.

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
        """
        return collate(
            samples,
            pad_idx=self.src_dict.pad(),
            eos_idx=self.eos,
            left_pad_source=self.left_pad_source,
            left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding, sentence_prediction=self.sentence_prediction,
            sentence_prediction_pool_target_eos=self.sentence_prediction_pool_target_eos,
            sentence_prediction_pool_source_eos=self.sentence_prediction_pool_source_eos,
            use_encoder_features=self.use_encoder_features, sentence_prediction_binary=self.sentence_prediction_binary, 
            sentence_regression=self.sentence_regression, every_timestep=self.every_timestep
        )

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        if self.buckets is None:
            # sort by target length, then source length
            if self.tgt_sizes is not None:
                indices = indices[
                    np.argsort(self.tgt_sizes[indices], kind='mergesort')
                ]
            return indices[np.argsort(self.src_sizes[indices], kind='mergesort')]
        else:
            # sort by bucketed_num_tokens, which is:
            #   max(padded_src_len, padded_tgt_len)
            return indices[
                np.argsort(self.bucketed_num_tokens[indices], kind='mergesort')
            ]

    @property
    def supports_prefetch(self):
        return (
            getattr(self.src, 'supports_prefetch', False)
            and (getattr(self.tgt, 'supports_prefetch', False) or self.tgt is None)
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        if self.tgt is not None:
            self.tgt.prefetch(indices)
        if self.align_dataset is not None:
            self.align_dataset.prefetch(indices)
