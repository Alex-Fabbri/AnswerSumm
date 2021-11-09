# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.criterions.span_loss import get_span_and_nll_loss


import sys
import time
import torch
from fairseq.data.data_utils import collate_tokens
from scipy.special import softmax


from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rougeLsum'], use_stemmer=True)

#from fast_bleu import BLEU, SelfBLEU

import numpy as np
import pickle as pkl
from sentence_transformers import SentenceTransformer
import spacy 
nlp = spacy.load("en_core_web_sm")

pca_fit = pkl.load(open("~/fairseq/pca.pkl", "rb"))
sentence_transformers_model = SentenceTransformer('bert-large-nli-stsb-mean-tokens')
sentence_transformers_model.eval()


def CalculateArea(arr):
    return 0.5 * np.abs(np.dot(arr[:, 0], np.roll(arr[:, 1], 1)) - np.dot(arr[:, 1], np.roll(arr[:, 0], 1)))

def clockwise(arr):
    if not type(arr) == list:
        arr2 = np.ndarray.tolist(arr)

    cent = (sum([p[0] for p in arr2])/len(arr2),sum([p[1] for p in arr2])/len(arr2))
    arr2.sort(key=lambda p: math.atan2(p[1]-cent[1],p[0]-cent[0]))

    return np.array(arr2)

def PolyVol(arr,r):
    r = 2
    if r == 2:
        polygon = clockwise(arr)
        return CalculateArea(polygon)

    elif r == 3:
        if len(arr) < r + 1:
            return 0
        else:
            delaunay = Delaunay(arr)
            vol = 0
            for i in delaunay.simplices:
                vol += PyramidVol(arr[i])
                return vol


def getscore(model, source_string, target_string, gold_target, loss_index=0):
    try:
        #source_string_sents =  source_string.split("<Q>")[1].split("<S>")
        source_string_sents = [x.text for x in nlp(source_string.split("</s>")[1]).sents]
    except:
        source_string_sents = source_string.split("<S>")
    # take at most 128 sentences from the source
    source_string_sents = [x.replace("<A>", " ").replace("<C>", "").replace("<Q>", "").replace("<m>", "").replace("</m>", "").strip() for x in source_string_sents if (not x.isspace() and len(x) > 0)][:128]
    source_string_sents = [x for x in source_string_sents if len(x.split()) > 3]
    target_sents = target_string.split("<S>")
    target_sents = [x.replace("<A>", "").replace("<C>", "").replace("<Q>", "").replace("<m>", "").replace("</m>", "").strip() for x in target_sents if not x.isspace() and len(x) > 0]
    if target_string.count("involved") > 5:
        return -1.0
    if target_string.count("!") > 5:
        return -1.0
    if len(target_sents) == 0:
        return 0.0
    if len(target_sents) > 10:
        return -1.0

    selfbleu_loss = False
    volume_loss = False
    rouge_loss = False
    nli_loss = False
    if loss_index == 0:
        nli_loss = True
    elif loss_index == 1:
        rouge_loss = True
    elif loss_index == 2:
        volume_loss = True
    elif loss_index == 3:
        selfbleu_loss = True

    sentence_level = getattr(model.args, 'nli_reinforce_sentence_level', False)
    # if we specified to only do one of these losses then ignore the indexing
    if getattr(model.args, 'nli_reinforce_selfbleu_only', False) or \
        getattr(model.args, 'nli_reinforce_volume_only', False) or \
            getattr(model.args, 'nli_reinforce_rouge_only', False) or \
                getattr(model.args, 'nli_reinforce_nli_only', False):
                selfbleu_loss = False
                volume_loss = False
                rouge_loss = False
                nli_loss = False
    
    # Diversity/selfbleu
    if getattr(model.args, 'nli_reinforce_selfbleu_only', False) or selfbleu_loss:
        bleu_object = SelfBLEU(target_sents)
        diversity_scores = bleu_object.get_score()[4]
        diversity_scores = [1 - x for x in diversity_scores]
        if sentence_level:
            return diversity_scores
        diversity_avg = sum(diversity_scores)/len(diversity_scores)
        return diversity_avg
    
    # Coverage/diversity/volume
    if getattr(model.args, 'nli_reinforce_volume_only', False) or volume_loss:
        target_sents_embeddings = sentence_transformers_model.encode(target_sents, show_progress_bar=False)
        target_sents_pca = pca_fit.transform(target_sents_embeddings) 
        if getattr(model.args, 'volume_no_sent_avg', False):
            vol = PolyVol(target_sents_pca, 2)
            vol_final = (vol - 0)/(111 - 0)
            vol_final = min(vol_final, 1)
            return vol_final
        else:
            vol = PolyVol(target_sents_pca, 2) #/len(target_sents)
            # 18
            # vol_final = (vol - 0)/(18 - 0)
            vol_final = (vol - 0)/(111 - 0)
            vol_final = min(vol_final, 1)
            return vol_final
    # ROUGE
    if getattr(model.args, 'nli_reinforce_rouge_only', False) or rouge_loss:
        gold_ref = "\n".join(gold_target.split("<S>"))
        if sentence_level:
            scores_to_return = [scorer.score(gold_ref, cur_sent)['rougeLsum'].fmeasure for cur_sent in target_sents]
            return scores_to_return
        scores_ = scorer.score(gold_ref, "\n".join(target_sents))
        rouge_score = scores_['rougeLsum'].fmeasure
        return rouge_score

    # NLI
    if getattr(model.args, 'nli_reinforce_nli_only', False) or nli_loss:
        scores = []
        slines = []
        ex_count = 0
        bsz = 128
        for target_sent in target_sents:
            for source_sent in source_string_sents:
                if ex_count % bsz == 0 and ex_count != 0:
                    with torch.no_grad():
                        batch = []
                        for pair in slines:
                            enc_0 = model.roberta.encode(pair[0]).tolist()
                            enc_1 = model.roberta.encode(pair[1]).tolist()
                            enc_0_len = 512 - len(enc_1)
                            if len(enc_1) >= 512:
                                enc_1 = enc_1[:256]
                                enc_0_len = 512 - len(enc_1)
                            if enc_0_len < len(enc_0):
                                enc_0_len -= 1
                            if enc_0_len < len(enc_0):
                                batch.append(torch.tensor(enc_0[:enc_0_len] + [2, 2] + enc_1[1:], dtype=torch.int64))
                            else:
                                batch.append(torch.tensor(enc_0[:enc_0_len] + [2] + enc_1[1:], dtype=torch.int64))
                        batch = collate_tokens(batch, pad_idx=1)
                        logprobs = model.roberta.predict('mnli', batch)
                        for prob in logprobs:
                            out = list(softmax(prob.cpu().numpy()))
                            scores.append(out[-1])
                    slines = []
                slines.append((source_sent, target_sent))
                ex_count += 1
        if slines != []:
            with torch.no_grad():
                batch = []
                for pair in slines:
                    enc_0 = model.roberta.encode(pair[0]).tolist()
                    enc_1 = model.roberta.encode(pair[1]).tolist()
                    enc_0_len = 512 - len(enc_1)
                    if len(enc_1) >= 512:
                        enc_1 = enc_1[:256]
                        enc_0_len = 512 - len(enc_1)
                    if enc_0_len < len(enc_0):
                        enc_0_len -= 1
                    if enc_0_len < len(enc_0):
                        batch.append(torch.tensor(enc_0[:enc_0_len] + [2, 2] + enc_1[1:], dtype=torch.int64))
                    else:
                        batch.append(torch.tensor(enc_0[:enc_0_len] + [2] + enc_1[1:], dtype=torch.int64))
                batch = collate_tokens(batch, pad_idx=1)
                logprobs = model.roberta.predict('mnli', batch)
                for prob in logprobs:
                    out = list(softmax(prob.cpu().numpy()))
                    scores.append(out[-1])
        sent_scores = []
        for j in range(len(target_sents)):
            try:
                cur_score = max(scores[j * len(source_string_sents): (j+1) * len(source_string_sents)])
            except:
                cur_score = 0.0
            sent_scores.append(cur_score)
        if sentence_level:
            return sent_scores
        nli_score = sum(sent_scores)/len(sent_scores)
        return nli_score
    # total_reward = nli_score + rouge_score
    # return total_reward


@register_criterion('span_rl')
class LabelSmoothedCrossEntropyCriterionSpanRL(FairseqCriterion):

    def __init__(self, task, sentence_avg, label_smoothing):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        # fmt: on

    def forward(self, model, sample, reduce=True, loss_index=0, nll_only=False, span_only=False):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss, nll_loss, span_loss = self.compute_loss(model, net_output, sample, reduce=reduce, loss_index=loss_index, nll_only=nll_only, span_only=span_only)
        # sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        if getattr(model.args, 'sample_size_one', False):
            sample_size = 1
        else:
            sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': loss.data,
            'nll_loss': nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        if span_loss is not None:
            logging_output['span_loss'] = span_loss.data

        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True, loss_index=0, nll_only=False, span_only=False):
        lprobs = None
        # reinforce run
        if getattr(model.args, 'nli_reinforce', False):
            if (not getattr(model.args,  'nli_reinforce_only', False)) or nll_only:
                if (not nll_only) and ((getattr(model.args,  'nli_reinforce_rotating_rl_only', False))):
                    pass
                else:
                    span_nll_loss, nll_loss, _ = get_span_and_nll_loss(model, net_output, sample, self.eps, self.padding_idx, reduce=reduce, nll_only=nll_only, span_only=span_only)
                    # if nll_only:
                    #     return span_nll_loss, nll_loss, None
                    if not model.training:
                      return span_nll_loss, nll_loss, None
                    #lprobs = model.get_normalized_probs(net_output, log_probs=True)
                    #lprobs = lprobs.view(-1, lprobs.size(-1))
                    #target = model.get_targets(sample, net_output).view(-1, 1)
                    #loss, nll_loss = label_smoothed_nll_loss(
                    #    lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
                    #)
                    #loss /= sample['ntokens']
                    #nll_loss /= sample['ntokens']
                    #if validating:
                    #    return loss, nll_loss, None
            # else:
                # loss = torch.tensor(0)
                # nll_loss = torch.tensor(0)
            if not getattr(model.args,  'nli_reinforce_lambda', False):
                nli_reinforce_lambda = 1.0
            else:
                nli_reinforce_lambda = model.args.nli_reinforce_lambda
            if not getattr(model.args, 'nli_reinforce_nll_lambda', False):
                nli_reinforce_nll_lambda = 1.0
            else:
                nli_reinforce_nll_lambda = model.args.nli_reinforce_nll_lambda

            logits = net_output[0].float()
            m = torch.distributions.Categorical(logits=logits)
            action = m.sample()
            action_greedy = logits.argmax(dim=-1)
            # lprobs_before_softmax = model.get_normalized_probs(net_output, log_probs=False)
            # m = torch.distributions.Categorical(lprobs_before_softmax)
            # action = m.sample()
            # action_greedy = lprobs_before_softmax.argmax(dim=-1)
            all_losses = []
            pad_mask = model.get_targets(sample, net_output).eq(self.padding_idx)
            for i in range(sample['net_input']['src_tokens'].shape[0]):
                cur_action = action[i, :]
                cur_action_greedy = action_greedy[i, :]

                string_action = self.task.source_dictionary.string(cur_action).split()
                string_action = " ".join([x for x in string_action if x.isdigit()])
                string_action = model.bpe.decode(string_action)

                string_greedy = self.task.source_dictionary.string(cur_action_greedy).split()
                string_greedy = " ".join([x for x in string_greedy if x.isdigit()])
                string_greedy = model.bpe.decode(string_greedy)

                string_source = self.task.source_dictionary.string(sample['net_input']['src_tokens'][i, :]).replace("<pad>", "").split()
                string_source = " ".join([x for x in string_source if x.isdigit()])
                string_source = model.bpe.decode(string_source)

                string_target = self.task.source_dictionary.string(model.get_targets(sample, net_output)[i, :]).replace("<pad>", "").split()
                string_target = " ".join([x for x in string_target if x.isdigit()])
                string_target = model.bpe.decode(string_target)
                if getattr(model.args, 'nli_pdb', False):
                    print(string_action)
                    from fairseq import pdb;pdb.set_trace()
                sampling_score = getscore(model, string_source, string_action, string_target, loss_index=loss_index)
                if getattr(model.args, 'nli_reinforce_no_baseline', False):
                    cur_loss = - sampling_score
                else:
                    baseline_score = getscore(model, string_source, string_greedy, string_target, loss_index=loss_index)
                    cur_loss = (baseline_score - sampling_score)
                # cur_loss = (baseline_score - sampling_score) * m.log_prob(action)
                # cur_mask = pad_mask[i, :]
                # cur_loss.masked_fill_(cur_mask, 0.)
                # cur_loss = cur_loss.sum()
                all_losses.append(cur_loss)

            log_prob = m.log_prob(action)
            log_prob.masked_fill_(pad_mask, 0.)
            final_rl_loss = []
            for cur_reward_loss, cur_log_prob in zip(all_losses, log_prob):
                final_rl_loss.append(cur_reward_loss * cur_log_prob.sum())
            rl_loss = sum(final_rl_loss)
            if getattr(model.args, 'nli_reinforce_normalize_sent', False):
                rl_loss /= sample['net_input']['src_tokens'].shape[0]
            else:
                rl_loss /= sample['ntokens']

            # Check reward is going to gradient
            # model.encoder.embed_tokens.weight.grad -> none
            # rl_loss.backward()
            # model.encoder.embed_tokens.weight.grad -> non zero
            if getattr(model.args,  'nli_reinforce_only', False) or \
                (getattr(model.args,  'nli_reinforce_rotating_rl_only', False) and (not nll_only)):
                return rl_loss, rl_loss, rl_loss
            else:
                # This lambda gets taken care of in the get_span_and_nll_loss function
                #total_loss = nli_reinforce_nll_lambda * loss + nli_reinforce_lambda * rl_loss
                total_loss =  span_nll_loss + nli_reinforce_lambda * rl_loss
                return total_loss, nll_loss, rl_loss
        # not a reinforce run - just reduces to span + nll loss
        else:
            span_nll_loss, nll_loss, _ = get_span_and_nll_loss(model, net_output, sample, self.eps, self.padding_idx, reduce=reduce, nll_only=nll_only, span_only=span_only)
            return span_nll_loss, nll_loss, None

    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in logging_outputs)
        span_loss_sum = sum(log.get('span_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        
        # in this case we probably have sample_size = 1
        if sample_size != ntokens:
            ntokens = sample_size


        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_scalar('span_loss', span_loss_sum / sample_size / math.log(2), ntokens, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
