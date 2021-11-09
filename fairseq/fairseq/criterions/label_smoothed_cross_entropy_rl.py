# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import label_smoothed_nll_loss
from fairseq.criterions.span_loss_rl import getscore # TODO running without this bc otherwise it gets copied twice (need to refactor code better)

import sys
import time
import torch
import numpy as np
import pickle as pkl
from fairseq.data.data_utils import collate_tokens
from scipy.special import softmax


@register_criterion('label_smoothed_cross_entropy_rl')
class LabelSmoothedCrossEntropyCriterionRL(FairseqCriterion):

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
        loss, nll_loss, span_loss = self.compute_loss(model, net_output, sample, reduce=reduce, loss_index=loss_index, nll_only=nll_only)
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

    def compute_loss(self, model, net_output, sample, reduce=True, loss_index=0, nll_only=False):
        lprobs = None
        if getattr(model.args, 'nli_reinforce', False):
            if (not getattr(model.args,  'nli_reinforce_only', False)) or nll_only:
                if (not nll_only) and ((getattr(model.args,  'nli_reinforce_rotating_rl_only', False))):
                    pass
                else:
                    lprobs = model.get_normalized_probs(net_output, log_probs=True)
                    lprobs = lprobs.view(-1, lprobs.size(-1))
                    target = model.get_targets(sample, net_output).view(-1, 1)
                    loss, nll_loss = label_smoothed_nll_loss(
                        lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
                    )
                    loss /= sample['ntokens']
                    nll_loss /= sample['ntokens']
                    if nll_only:
                        return loss, nll_loss, None
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
                (getattr(model.args,  'nli_reinforce_rotating_rl_only', False) and (not validating)):
                return rl_loss, rl_loss, rl_loss
            else:
                total_loss = nli_reinforce_nll_lambda * loss + nli_reinforce_lambda * rl_loss
                return total_loss, nll_loss, rl_loss
                
        else:
            lprobs = model.get_normalized_probs(net_output, log_probs=True)
            lprobs = lprobs.view(-1, lprobs.size(-1))
            target = model.get_targets(sample, net_output).view(-1, 1)
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
            )
            return loss, nll_loss, None

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
