# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List
from scipy import spatial
from fairseq import utils
from fairseq.data import encoders
from typing import Dict, List, Optional
from torch import Tensor
import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from fairseq import search, utils
from fairseq.data import data_utils
from fairseq.models import FairseqIncrementalDecoder
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.sequence_generator import BeamContainer

logger = logging.getLogger(__name__)


class BARTHubInterface(nn.Module):
    """A simple PyTorch Hub interface to BART.

    Usage: https://github.com/pytorch/fairseq/tree/master/examples/BART
    """

    def __init__(self, args, task, model):
        super().__init__()
        self.args = args
        self.task = task
        self.model = model

        self.bpe = encoders.build_bpe(args)

        self.max_positions = min(utils.resolve_max_positions(
            self.task.max_positions(),
            self.model.max_positions(),
        ))

        # this is useful for determining the device
        self.register_buffer('_float_tensor', torch.tensor([0], dtype=torch.float))

    @property
    def device(self):
        return self._float_tensor.device

    def encode(self, sentence: str, *addl_sentences, no_separator=True) -> torch.LongTensor:
        """
        BPE-encode a sentence (or multiple sentences).

        Every sequence begins with a beginning-of-sentence (`<s>`) symbol.
        Every sentence ends with an end-of-sentence (`</s>`).

        Example (single sentence): `<s> a b c </s>`
        Example (sentence pair): `<s> d e f </s> 1 2 3 </s>`

        The BPE encoding follows GPT-2. One subtle detail is that the GPT-2 BPE
        requires leading spaces. For example::

            >>> bart.encode('Hello world').tolist()
            [0, 31414, 232, 2]
            >>> bart.encode(' world').tolist()
            [0, 232, 2]
            >>> bart.encode('world').tolist()
            [0, 8331, 2]
        """
        # tokens = self.bpe.encode(sentence)
        # # TODO(alex-fabbri): I fine-tuned without bos - # https://github.com/pytorch/fairseq/issues/1389#issuecomment-568672905
        # # if len(tokens.split(' ')) > self.max_positions - 2:
        # #     tokens = ' '.join(tokens.split(' ')[:self.max_positions - 2])
        # if len(tokens.split(' ')) > self.max_positions - 1:
        #     tokens = ' '.join(tokens.split(' ')[:self.max_positions - 1])
        # bpe_sentence = tokens + ' </s>'
        # for s in addl_sentences:
        #     bpe_sentence += (' </s>' if not no_separator else '')
        #     bpe_sentence += ' ' + self.bpe.encode(s) + ' </s>'
        # tokens = self.task.source_dictionary.encode_line(bpe_sentence, append_eos=False)
        # return tokens.long()
        tokens = self.bpe.encode(sentence)
        if len(tokens.split(' ')) > self.max_positions - 2:
            tokens = ' '.join(tokens.split(' ')[:self.max_positions - 2])
        bpe_sentence = '<s> ' + tokens + ' </s>'
        for s in addl_sentences:
            bpe_sentence += (' </s>' if not no_separator else '')
            bpe_sentence += ' ' + self.bpe.encode(s) + ' </s>'
        tokens = self.task.source_dictionary.encode_line(bpe_sentence, append_eos=False)
        return tokens.long()

    def decode(self, tokens: torch.LongTensor):
        assert tokens.dim() == 1
        tokens = tokens.cpu().numpy()
        if tokens[0] == self.task.source_dictionary.bos():
            tokens = tokens[1:]  # remove <s>
        eos_mask = (tokens == self.task.source_dictionary.eos())
        doc_mask = eos_mask[1:] & eos_mask[:-1]
        sentences = np.split(tokens, doc_mask.nonzero()[0] + 1)
        sentences = [self.bpe.decode(self.task.source_dictionary.string(s)) for s in sentences]
        if len(sentences) == 1:
            return sentences[0]
        return sentences

    def _build_sample(self, src_tokens: List[torch.LongTensor]):
        # assert torch.is_tensor(src_tokens)
        dataset = self.task.build_dataset_for_inference(
            src_tokens,
            [x.numel() for x in src_tokens],
        )
        sample = dataset.collater(dataset)
        sample = utils.apply_to_sample(
            lambda tensor: tensor.to(self.device),
            sample
        )
        return sample

    def _build_sample_span(self, src_tokens: List[torch.LongTensor], tgt_tokens: List[torch.LongTensor]):
        # assert torch.is_tensor(src_tokens)
        dataset = self.task.build_dataset_for_inference(
            src_tokens,
            [x.numel() for x in src_tokens],
            tgt_tokens, 

        )
        sample = dataset.collater(dataset)
        sample = utils.apply_to_sample(
            lambda tensor: tensor.to(self.device),
            sample
        )
        return sample

    def sample(self, sentences: List[str], beam: int = 1, verbose: bool = False, **kwargs) -> str:
        input = [self.encode(sentence) for sentence in sentences]
        hypos = self.generate(input, beam, verbose, **kwargs)
        return [self.decode(x['tokens']) for x in hypos]

    def sample_iterate(self, sentences: List[str], beam: int = 1, verbose: bool = False, **kwargs) -> str:
        input = [self.encode(sentence) for sentence in sentences]
        hypos = self.generate_iterate(input, beam, verbose, **kwargs)
        return [self.decode(x['tokens']) for x in hypos]
    
    def sample_spans(self, sentences: List[str], beam: int = 1, verbose: bool = False, **kwargs) -> str:
        input_src = [self.encode(sentence[0]) for sentence in sentences]
        input_tgt = [self.encode(sentence[1]) for sentence in sentences]
        sample = self._build_sample_span(input_src, input_tgt)
        net_output = self.model(**sample["net_input"])

        # import time
        # start = time.time()
        start_argmax = torch.argmax(net_output[1]['start_logits'], dim=1)
        end_argmax = torch.argmax(net_output[1]['end_logits'], dim=1)
        sent_ids = sample['net_input']['target_sent_ids']
        # cur_tokens = sample['net_input']['prev_input_tokens']
        # TODO modify if I used 
        cur_tokens = sample['net_input']['src_tokens']
        all_examples = []
        for count, (cur_token, sent_id, start_, end_) in enumerate(zip(cur_tokens, sent_ids, start_argmax, end_argmax)):
            # if count == 13:
            #     from fairseq import pdb;pdb.set_trace()
            # tmp_cur_token = cur_token
            # tmp_cur_token[cur_token == 1] = 2
            # tmp_out = " ".join(self.decode(tmp_cur_token))
            # if "ou can contact the owner of the property about his " in tmp_out:
            #     from fairseq import pdb;pdb.set_trace()
            cur_example = []
            for cur_id, start, end in zip(sent_id, start_, end_):
                if cur_id == 1:
                    continue
                try:
                    #cur_example.append(self.decode(cur_token[start:end+1]).replace("<S>", "<sent>"))
                    # from fairseq import pdb;pdb.set_trace()
                    if start + cur_token.tolist()[start:].index(50257) == start:
                        end = start + cur_token.tolist()[start+1:].index(50257)
                    else:
                        end = start + cur_token.tolist()[start:].index(50257)
                    cur_str = self.decode(cur_token[start: end]).replace("<S>", "")
                    if len(cur_str) == 0 or cur_str.isspace():
                        cur_example.append("NONE")
                    else:
                        cur_example.append(cur_str)
                except:
                    cur_example.append("NONE")
            cur_example_str = " <S> ".join(cur_example)
            all_examples.append(cur_example_str)
        # end = time.time()
        # print(end - start)
        hypos = [v for _, v in sorted(zip(sample['id'].tolist(), all_examples))]
        return hypos

    def sample_sents_regression(self, sentences: List[str], beam: int = 1, verbose: bool = False, **kwargs) -> str:

        input_src = [self.encode(sentence[0]) for sentence in sentences]
        input_tgt = [self.encode(sentence[1]) for sentence in sentences]
        sample = self._build_sample_span(input_src, input_tgt)
        net_output = self.model(**sample["net_input"])

        # import time
        # start = time.time()
        try:
            sentence_prediction = net_output[1]['sentence_prediction']
        except:
            sentence_prediction = net_output[1]['attn_']
            sentence_prediction = sentence_prediction.transpose(1, 2)
        # sentence_prediction_binary = sentence_prediction[:, 1, :, :]
        summarization_ids_out = sample['net_input']['summarization_ids_out']
        # sentence_prediction_argmax = torch.argmax(sentence_prediction, dim=1)
        # start_argmax = torch.argmax(net_output[1]['start_logits'], dim=1)
        # end_argmax = torch.argmax(net_output[1]['end_logits'], dim=1)
        # sent_ids = sample['net_input']['target_sent_ids']
        target_sent_ids = sample['net_input']['target_sent_ids']
        cur_tokens = sample['net_input']['src_tokens']
        all_examples = []
        for cur_token, sent_id, sentence_pred, target_sent_id in zip(cur_tokens, summarization_ids_out, sentence_prediction, target_sent_ids):
            cur_example = []
            # try:
            #     largest_id = sent_id.tolist().index(1)
            # except:
            #     largest_id = len(sent_id)
            # sentence_pred_spans = sentence_pred[1, :, :largest_id]
            # sentence_pred_spans_pred = sentence_pred.argmax(dim=0)
            if getattr(self.args, "sentence_prediction_pool_source_eos", False):
                # sent_id_orig = sentence_pred_spans_pred
                sent_id = [sent_id[i+1] for i in range(0, len(sent_id)-1, 2)]
            # from fairseq import pdb;pdb.set_trace()
            sentence_pred = sentence_pred.t()
            for sent_pred, target_id in zip(sentence_pred, target_sent_id):
                if target_id == 1:
                    continue
                try:
                    sent_pred_max_index = sent_id.tolist().index(1)
                    sent_pred = sent_pred[:sent_pred_max_index]
                except:
                    sent_pred = sent_pred[:]
                span = sent_pred.argmax(dim=0)
                try:
                    if span == 0:
                        # start = 0
                        start = cur_token.tolist().index(50257)
                        end = sent_id[span]
                    else:
                        start = sent_id[span-1]
                        end = sent_id[span]
                    if getattr(self.args, "sentence_prediction_pool_source_eos", False):
                        out = self.decode(cur_token[start+1:end]).replace("<S>", " ")
                    else:
                        out = self.decode(cur_token[start+1:end+1]).replace("<S>", " ")
                    cur_example.append(out) # start+1 since the ids are actually for the end of sentence
                    #  -- there may be tokens in between but can be postprocessed
                except:
                    # from fairseq import pdb;pdb.set_trace()
                    cur_example.append("NONE")
            cur_example_str = " <S> ".join(cur_example)
            all_examples.append(cur_example_str)
        # end = time.time()
        # print(end - start)
        hypos = [v for _, v in sorted(zip(sample['id'].tolist(), all_examples))]
        return hypos
    
    def sample_sents(self, sentences: List[str], beam: int = 1, verbose: bool = False, **kwargs) -> str:
        # from fairseq import pdb;pdb.set_trace()
        input_src = [self.encode(sentence[0]) for sentence in sentences]
        input_tgt = [self.encode(sentence[1]) for sentence in sentences]
        sample = self._build_sample_span(input_src, input_tgt)
        net_output = self.model(**sample["net_input"])

        # import time
        # start = time.time()
        try:
            sentence_prediction = net_output[1]['sentence_prediction']
        except:
            sentence_prediction = net_output[1]['attn_']
            sentence_prediction = sentence_prediction.transpose(1, 2)
        summarization_ids_out = sample['net_input']['summarization_ids_out']
        # sentence_prediction_argmax = torch.argmax(sentence_prediction, dim=1)
        # start_argmax = torch.argmax(net_output[1]['start_logits'], dim=1)
        # end_argmax = torch.argmax(net_output[1]['end_logits'], dim=1)
        # sent_ids = sample['net_input']['target_sent_ids']
        target_sent_ids = sample['net_input']['target_sent_ids']
        cur_tokens = sample['net_input']['src_tokens']
        all_examples = []
        for cur_token, sent_id, sentence_pred, target_sent_id in zip(cur_tokens, summarization_ids_out, sentence_prediction, target_sent_ids):
            cur_example = []
            # try:
            #     largest_id = sent_id.tolist().index(1)
            # except:
            #     largest_id = len(sent_id)
            # sentence_pred_spans = sentence_pred[1, :, :largest_id]
            sentence_pred_spans_pred = sentence_pred.argmax(dim=0)
            if getattr(self.args, "sentence_prediction_pool_source_eos", False):
                # sent_id_orig = sentence_pred_spans_pred
                sent_id = [sent_id[i+1] for i in range(0, len(sent_id)-1, 2)]
            if getattr(self.args, "sentence_prediction_pool_target_eos", False):
                target_sent_id = target_sent_id[1:]
            for span, target_id in zip(sentence_pred_spans_pred, target_sent_id):
                if target_id == 1:
                    continue
                try:
                    if span == 0:
                        # start = 0
                        start = cur_token.tolist().index(50257)
                        end = sent_id[span]
                    else:
                        start = sent_id[span-1]
                        end = sent_id[span]
                    if getattr(self.args, "sentence_prediction_pool_source_eos", False):
                        out = self.decode(cur_token[start+1:end]).replace("<S>", " ")
                    else:
                        out = self.decode(cur_token[start+1:end+1]).replace("<S>", " ")
                    cur_example.append(out) # start+1 since the ids are actually for the end of sentence
                    #  -- there may be tokens in between but can be postprocessed
                except:
                    #from fairseq import pdb;pdb.set_trace()
                    cur_example.append("NONE")
            cur_example_str = " <S> ".join(cur_example)
            all_examples.append(cur_example_str)
        # end = time.time()
        # print(end - start)
        hypos = [v for _, v in sorted(zip(sample['id'].tolist(), all_examples))]
        return hypos

    def generate(self, tokens: List[torch.LongTensor], beam: int = 5, verbose: bool = False, **kwargs) -> torch.LongTensor:
        sample = self._build_sample(tokens)

        # build generator using current args as well as any kwargs
        gen_args = copy.copy(self.args)
        gen_args.beam = beam
        for k, v in kwargs.items():
            setattr(gen_args, k, v)
        generator = self.task.build_generator([self.model], gen_args)
        translations = self.task.inference_step(
            generator,
            [self.model],
            sample,
            prefix_tokens=sample['net_input']['src_tokens'].new_zeros((len(tokens), 1)).fill_(self.task.source_dictionary.bos()),
            bos_token=self.task.source_dictionary.bos(),
        )

        if verbose:
            src_str_with_unk = self.string(tokens)
            logger.info('S\t{}'.format(src_str_with_unk))

        def getarg(name, default):
            return getattr(gen_args, name, getattr(self.args, name, default))

        # Process top predictions
        hypos = [x[0] for x in translations]
        hypos = [v for _, v in sorted(zip(sample['id'].tolist(), hypos))]
        return hypos

    def generate_iterate(self, tokens: List[torch.LongTensor], beam: int = 5, verbose: bool = False, **kwargs) -> torch.LongTensor:
        sample = self._build_sample(tokens)

        # build generator using current args as well as any kwargs
        gen_args = copy.copy(self.args)
        gen_args.beam = beam
        for k, v in kwargs.items():
            setattr(gen_args, k, v)
        generator = self.task.build_generator([self.model], gen_args)
        # generator._generate()
        # translations = self.task.inference_step(
        #     generator,
        #     [self.model],
        #     sample,
        #     prefix_tokens=sample['net_input']['src_tokens'].new_zeros((len(tokens), 1)).fill_(self.task.source_dictionary.bos()),
        #     ,
        # )
        # from fairseq import pdb;pdb.set_trace()
        bos_token=self.task.source_dictionary.bos()
        incremental_states = torch.jit.annotate(
            List[Dict[str, Dict[str, Optional[Tensor]]]],
            [
                torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {})
                for i in range(generator.model.models_size)
            ],
        )
        net_input = sample["net_input"]
        src_tokens = net_input["src_tokens"]
        # length of the source text being the character length except EndOfSentence and pad
        src_lengths = (
            (src_tokens.ne(generator.eos) & src_tokens.ne(generator.pad)).long().sum(dim=1)
        )
        # bsz: total number of sentences in beam
        input_size = src_tokens.size()
        bsz, src_len = input_size[0], input_size[1]
        beam_size = generator.beam_size

        max_len: int = -1
        if generator.match_source_len:
            max_len = src_lengths.max().item()
        else:
            max_len = min(
                int(generator.max_len_a * src_len + generator.max_len_b),
                # exclude the EOS marker
                generator.model.max_decoder_positions() - 1,
            )
        assert (
            generator.min_len <= max_len
        ), "min_len cannot be larger than max_len, please adjust these!"
        # compute the encoder output for each beam
        encoder_outs = generator.model.forward_encoder(net_input)

        # placeholder of indices for bsz * beam_size to hold tokens and accumulative scores
        new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
        new_order = new_order.to(src_tokens.device).long()
        encoder_outs = generator.model.reorder_encoder_out(encoder_outs, new_order)
        # ensure encoder_outs is a List.
        assert encoder_outs is not None

        # initialize buffers
        scores = (
            torch.zeros(bsz * beam_size, max_len + 1).to(src_tokens).float()
        )  # +1 for eos; pad is never choosed for scoring
        tokens = (
            torch.zeros(bsz * beam_size, max_len + 2)
            .to(src_tokens)
            .long()
            .fill_(generator.pad)
        )  # +2 for eos and pad
        tokens[:, 0] = generator.eos if bos_token is None else bos_token
        attn: Optional[Tensor] = None

        # A list that indicates candidates that should be ignored.
        # For example, suppose we're sampling and have already finalized 2/5
        # samples. Then cands_to_ignore would mark 2 positions as being ignored,
        # so that we only finalize the remaining 3 samples.
        cands_to_ignore = (
            torch.zeros(bsz, beam_size).to(src_tokens).eq(-1)
        )  # forward and backward-compatible False mask

        # list of completed sentences
        finalized = torch.jit.annotate(
            List[List[Dict[str, Tensor]]],
            [torch.jit.annotate(List[Dict[str, Tensor]], []) for i in range(bsz)],
        )  # contains lists of dictionaries of infomation about the hypothesis being finalized at each step

        finished = [
            False for i in range(bsz)
        ]  # a boolean array indicating if the sentence at the index is finished or not
        num_remaining_sent = bsz  # number of sentences remaining

        # number of candidate hypos per step
        cand_size = 2 * beam_size  # 2 x beam size in case half are EOS

        # offset arrays for converting between different indexing schemes
        bbsz_offsets = (torch.arange(0, bsz) * beam_size).unsqueeze(1).type_as(tokens)
        cand_offsets = torch.arange(0, cand_size).type_as(tokens)

        reorder_state: Optional[Tensor] = None
        batch_idxs: Optional[Tensor] = None
        prev_stopper = 0
        for step in range(max_len + 1):  # one extra step for EOS marker
            # reorder decoder internal states based on the prev choice of beams
            # print(f'step: {step}')
            if reorder_state is not None:
                if batch_idxs is not None:
                    # update beam indices to take into account removed sentences
                    corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(
                        batch_idxs
                    )
                    reorder_state.view(-1, beam_size).add_(
                        corr.unsqueeze(-1) * beam_size
                    )
                generator.model.reorder_incremental_state(incremental_states, reorder_state)
                encoder_outs = generator.model.reorder_encoder_out(
                    encoder_outs, reorder_state
                )
            lprobs, avg_attn_scores, decoder_features = generator.model.forward_decoder(
                tokens[:, : step + 1],
                encoder_outs,
                incremental_states,
                generator.temperature,
                features_in_extra=True,
            )
            
            lprobs[lprobs != lprobs] = torch.tensor(-math.inf).to(lprobs)

            lprobs[:, generator.pad] = -math.inf  # never select pad
            lprobs[:, generator.unk] -= generator.unk_penalty  # apply unk penalty

            # handle max length constraint
            if step >= max_len:
                lprobs[:, : generator.eos] = -math.inf
                lprobs[:, generator.eos + 1 :] = -math.inf

            # handle prefix tokens (possibly with different lengths)
            prefix_tokens = None
            if (
                prefix_tokens is not None
                and step < prefix_tokens.size(1)
                and step < max_len
            ):
                lprobs, tokens, scores = generator._prefix_tokens(
                    step, lprobs, scores, tokens, prefix_tokens, beam_size
                )
            elif step < generator.min_len:
                # minimum length constraint (does not apply if using prefix_tokens)
                lprobs[:, generator.eos] = -math.inf

            # Record attention scores, only support avg_attn_scores is a Tensor
            if avg_attn_scores is not None:
                if attn is None:
                    attn = torch.empty(
                        bsz * beam_size, avg_attn_scores.size(1), max_len + 2
                    ).to(scores)
                attn[:, :, step + 1].copy_(avg_attn_scores)

            scores = scores.type_as(lprobs)
            eos_bbsz_idx = torch.empty(0).to(
                tokens
            )  # indices of hypothesis ending with eos (finished sentences)
            eos_scores = torch.empty(0).to(
                scores
            )  # scores of hypothesis ending with eos (finished sentences)

            if generator.should_set_src_lengths:
                generator.search.set_src_lengths(src_lengths)

            if generator.no_repeat_ngram_size > 0:
                lprobs = generator._no_repeat_ngram(tokens, lprobs, bsz, beam_size, step)

            cand_scores, cand_indices, cand_beams = generator.search.step(
                step,
                lprobs.view(bsz, -1, generator.vocab_size),
                scores.view(bsz, beam_size, -1)[:, :, :step],
            )
            # if cand_indices.detach().tolist() == [[50257]]:
            #     encoder_out = encoder_outs[0].encoder_out.squeeze() # src_len x 1024
            #     summarization_ids_out = (src_tokens[0]==50257).nonzero().squeeze().unsqueeze(-1)
            #     summarization_ids_out = summarization_ids_out.repeat(1, 1024)
            #     decoder_features = decoder_features.squeeze().cpu()
            #     encoder_features = torch.gather(encoder_out, 0, summarization_ids_out).cpu()
            #     max_sim = -1
            #     for ex in encoder_features:
            #         max_sim = max(max_sim, 1 - spatial.distance.cosine(ex, decoder_features))
            #     if max_sim < 0.5:
            #         tokens[0, prev_stopper:] = 1
            #         scores[0, prev_stopper:] = 0.0
    
            # cand_bbsz_idx contains beam indices for the top candidate
            # hypotheses, with a range of values: [0, bsz*beam_size),
            # and dimensions: [bsz, cand_size]
            cand_bbsz_idx = cand_beams.add(bbsz_offsets)

            # finalize hypotheses that end in eos
            eos_mask = cand_indices.eq(generator.eos) & cand_scores.ne(-math.inf)
            eos_mask[:, :beam_size][cands_to_ignore] = torch.tensor(0).to(eos_mask)

            # only consider eos when it's among the top beam_size indices
            eos_bbsz_idx = torch.masked_select(
                cand_bbsz_idx[:, :beam_size], mask=eos_mask[:, :beam_size]
            )

            finalized_sents: List[int] = []
            if eos_bbsz_idx.numel() > 0:
                eos_scores = torch.masked_select(
                    cand_scores[:, :beam_size], mask=eos_mask[:, :beam_size]
                )
                finalized_sents = generator.finalize_hypos(
                    step,
                    eos_bbsz_idx,
                    eos_scores,
                    tokens,
                    scores,
                    finalized,
                    finished,
                    beam_size,
                    attn,
                    src_lengths,
                    max_len,
                )
                
                num_remaining_sent -= len(finalized_sents)

            assert num_remaining_sent >= 0
            if num_remaining_sent == 0:
                break
            assert step < max_len

            if len(finalized_sents) > 0:
                new_bsz = bsz - len(finalized_sents)

                # construct batch_idxs which holds indices of batches to keep for the next pass
                batch_mask = torch.ones(bsz).to(cand_indices)
                batch_mask[
                    torch.tensor(finalized_sents).to(cand_indices)
                ] = torch.tensor(0).to(batch_mask)
                batch_idxs = batch_mask.nonzero().squeeze(-1)

                eos_mask = eos_mask[batch_idxs]
                cand_beams = cand_beams[batch_idxs]
                bbsz_offsets.resize_(new_bsz, 1)
                cand_bbsz_idx = cand_beams.add(bbsz_offsets)
                cand_scores = cand_scores[batch_idxs]
                cand_indices = cand_indices[batch_idxs]

                if prefix_tokens is not None:
                    prefix_tokens = prefix_tokens[batch_idxs]
                src_lengths = src_lengths[batch_idxs]
                cands_to_ignore = cands_to_ignore[batch_idxs]

                scores = scores.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                tokens = tokens.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                if attn is not None:
                    attn = attn.view(bsz, -1)[batch_idxs].view(
                        new_bsz * beam_size, attn.size(1), -1
                    )
                bsz = new_bsz
            else:
                batch_idxs = None
            # set active_mask so that values > cand_size indicate eos hypos
            # and values < cand_size indicate candidate active hypos.
            # After, the min values per row are the top candidate active hypos

            # Rewrite the operator since the element wise or is not supported in torchscript.

            eos_mask[:, :beam_size] = ~((~cands_to_ignore) & (~eos_mask[:, :beam_size]))
            active_mask = torch.add(
                eos_mask.type_as(cand_offsets) * cand_size,
                cand_offsets[: eos_mask.size(1)],
            )

            # get the top beam_size active hypotheses, which are just the hypos
            # with the smallest values in active_mask
            new_cands_to_ignore, active_hypos = torch.topk(
                active_mask, k=beam_size, dim=1, largest=False
            )

            # update cands_to_ignore to ignore any finalized hypos
            cands_to_ignore = new_cands_to_ignore.ge(cand_size)[:, :beam_size]
            assert (~cands_to_ignore).any(dim=1).all()

            active_bbsz_idx = torch.gather(cand_bbsz_idx, dim=1, index=active_hypos)
            active_scores = torch.gather(cand_scores, dim=1, index=active_hypos)

            active_bbsz_idx = active_bbsz_idx.view(-1)
            active_scores = active_scores.view(-1)

            # copy tokens and scores for active hypotheses
            tokens[:, : step + 1] = torch.index_select(
                tokens[:, : step + 1], dim=0, index=active_bbsz_idx
            )
            tokens.view(bsz, beam_size, -1)[:, :, step + 1] = torch.gather(
                cand_indices, dim=1, index=active_hypos
            )
            if step > 0:
                scores[:, :step] = torch.index_select(
                    scores[:, :step], dim=0, index=active_bbsz_idx
                )
            scores.view(bsz, beam_size, -1)[:, :, step] = torch.gather(
                cand_scores, dim=1, index=active_hypos
            )

            # copy attention for active hypotheses
            if attn is not None:
                attn[:, :, : step + 2] = torch.index_select(
                    attn[:, :, : step + 2], dim=0, index=active_bbsz_idx
                )

            # reorder incremental state in decoder
            reorder_state = active_bbsz_idx

        # sort by score descending
        for sent in range(len(finalized)):
            # make into beam container
            BCList = [
                BeamContainer(elem["score"].item(), elem) for elem in finalized[sent]
            ]
            BCList.sort()
            BCList.reverse()
            finalized[sent] = torch.jit.annotate(
                List[Dict[str, Tensor]], [x.elem for x in BCList]
            )

        translations = finalized

        if verbose:
            src_str_with_unk = generator.string(tokens)
            logger.info('S\t{}'.format(src_str_with_unk))

        def getarg(name, default):
            return getattr(gen_args, name, getattr(generator.args, name, default))

        # Process top predictions
        hypos = [x[0] for x in translations]
        hypos = [v for _, v in sorted(zip(sample['id'].tolist(), hypos))]
        return hypos

    def extract_features(self, tokens: torch.LongTensor, return_all_hiddens: bool = False) -> torch.Tensor:
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        if tokens.size(-1) > min(self.model.max_positions()):
            raise ValueError('tokens exceeds maximum length: {} > {}'.format(
                tokens.size(-1), self.model.max_positions()
            ))
        tokens.to(device=self.device),
        prev_output_tokens = tokens.clone()

        prev_output_tokens[:, 0] = tokens.gather(
            1,
            (tokens.ne(self.task.source_dictionary.pad()).sum(dim=1)- 1).unsqueeze(-1),
        ).squeeze()

        prev_output_tokens[:, 1:] = tokens[:, :-1]
        features, extra = self.model(
            src_tokens=tokens,
            src_lengths=None,
            prev_output_tokens=prev_output_tokens,
            features_only=True,
            return_all_hiddens=return_all_hiddens,
        )
        if return_all_hiddens:
            # convert from T x B x C -> B x T x C
            inner_states = extra['inner_states']
            return [inner_state.transpose(0, 1) for inner_state in inner_states]
        else:
            return features  # just the last layer's features

    def register_classification_head(
        self, name: str, num_classes: int = None, embedding_size: int = None, **kwargs
    ):
        self.model.register_classification_head(
            name, num_classes=num_classes, embedding_size=embedding_size, **kwargs
        )

    def predict(self, head: str, tokens: torch.LongTensor, return_logits: bool = False):
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        features = self.extract_features(tokens.to(device=self.device))
        sentence_representation = features[
            tokens.eq(self.task.source_dictionary.eos()), :
        ].view(features.size(0), -1, features.size(-1))[:, -1, :]

        logits = self.model.classification_heads[head](sentence_representation)
        if return_logits:
            return logits
        return F.log_softmax(logits, dim=-1)
