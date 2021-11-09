# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F


from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion

def get_span_and_nll_loss(model, net_output, sample, eps, padding_idx, reduce=True, nll_only=False, span_only=False):
    loss = 0.0
    nll_loss = 0.0
    if getattr(model.args, 'span_loss_lambda', False):
        span_lambda = float(model.args.span_loss_lambda)
    else:
        span_lambda = 1.0
    if getattr(model.args, 'nll_loss_lambda', False):
        nll_loss_lambda = float(model.args.nll_loss_lambda)
    else:
        nll_loss_lambda = 1.0
    if not span_only:
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, eps, ignore_index=padding_idx, reduce=reduce,
        )
        if getattr(model.args, 'sample_size_one', False):
            loss /= sample['ntokens']
            nll_loss /= sample['ntokens']
        loss *= nll_loss_lambda
        if nll_only:
            return loss, nll_loss, nll_loss
    if not getattr(model.args, 'use_span', False):
        return loss, nll_loss, nll_loss
    else:
        # 1) for dummy classifier which just tries to predict a 0 target label
        if getattr(model.args, 'dummy_classifier', False):
            logits = net_output[1]['classifier_logits']
            lprobs = F.log_softmax(logits, dim=1, dtype=torch.float32)
            classifier_targets = sample['dummy_targets']
            loss_dummy = F.nll_loss(lprobs, classifier_targets)
            total_loss = loss + loss_dummy
            return total_loss, nll_loss, total_loss

        elif getattr(model.args, 'simple_linear', False):
            # 2) for full span prediction loss
            # 'correct' loss with sample_size accounted for locally 
            start_positions = sample['span_start_ids']
            end_positions = sample['span_end_ids']
            start_logits = net_output[1]['start_logits']
            end_logits = net_output[1]['end_logits']

            # 
            # start_lprobs = F.log_softmax(start_logits, dim=1, dtype=torch.float32) # causes an error!
            #import pdb;pdb.set_trace()
            start_lprobs = F.log_softmax(start_logits, dim=1)
            start_loss = F.nll_loss(start_lprobs, start_positions, ignore_index=2000)
            # start_loss = self.loss_fct_linear(start_logits, start_positions)

            # end_lprobs = F.log_softmax(end_logits, dim=1, dtype=torch.float32)
            end_lprobs = F.log_softmax(end_logits, dim=1)
            end_loss = F.nll_loss(end_lprobs, end_positions, ignore_index=2000)
            # end_loss = self.loss_fct_linear(end_logits, end_positions)

            total_span_loss = (start_loss + end_loss) / 2   
            total_loss = loss + span_lambda * total_span_loss
            return total_loss, nll_loss, total_span_loss

        elif getattr(model.args, 'sentence_prediction', False):
            if getattr(model.args, 'sentence_prediction_binary', False):
                summarization_targets = sample['summarization_targets']
                sentence_prediction = net_output[1]['sentence_prediction']
                l_probs = F.log_softmax(sentence_prediction, dim=1)
                extractive_loss = F.nll_loss(l_probs, summarization_targets, ignore_index=2000) # hard code for now
                total_loss = span_lambda * extractive_loss + loss
                return total_loss, nll_loss, extractive_loss
            elif getattr(model.args, 'sentence_prediction_single_span', False) and not getattr(model.args, 'sentence_prediction_span_head_attn', False) :
                summarization_targets = sample['summarization_targets']
                sentence_prediction = net_output[1]['sentence_prediction']
                l_probs = F.log_softmax(sentence_prediction, dim=1)
                extractive_loss = F.nll_loss(l_probs, summarization_targets, ignore_index=2000) # hard code for now
                total_loss = span_lambda * extractive_loss + loss
                return total_loss, nll_loss, extractive_loss
            elif getattr(model.args, 'sentence_prediction_span_head_attn', False):
                attn_ = net_output[1]['attn_']
                attn_ = attn_.transpose(1, 2)
                summarization_targets = sample['summarization_targets']
                # summarization_targets_tmp = summarization_targets
                # summarization_targets = summarization_targets.unsqueeze(-1)
                # summarization_targets[summarization_targets==2000] = 0
                # lprobs = (torch.gather(attn_, 2, summarization_targets)).log()
                lprobs = attn_.log()
                extractive_loss = F.nll_loss(lprobs, summarization_targets, ignore_index=2000) # hard code for now
                # extractive_loss = -(torch.gather(attn_, 2, summarization_targets).log()).sum()
                total_loss = span_lambda * extractive_loss + loss
                # print(total_loss, nll_loss, extractive_loss)
                return total_loss, nll_loss, extractive_loss
                # l_probs = F.log_softmax(sentence_prediction, dim=1)
                # extractive_loss = F.nll_loss(l_probs, summarization_targets, ignore_index=1000) # hard code for now
                # total_loss = extractive_loss + loss
                # print(total_loss, nll_loss, extractive_loss)
                # return total_loss, nll_loss, extractive_loss
            elif getattr(model.args, 'sentence_prediction_attn_direct', False):
                attn_ = net_output[1]['attn_out']
                attn_ = attn_.transpose(1, 2)
                summarization_targets = sample['summarization_targets']
                # summarization_targets_tmp = summarization_targets
                # summarization_targets = summarization_targets.unsqueeze(-1)
                # summarization_targets[summarization_targets==2000] = 0
                # lprobs = (torch.gather(attn_, 2, summarization_targets)).log()
                lprobs = attn_.log()
                extractive_loss = F.nll_loss(lprobs, summarization_targets, ignore_index=2000) # hard code for now
                # extractive_loss = -(torch.gather(attn_, 2, summarization_targets).log()).sum()
                total_loss = span_lambda * extractive_loss + loss
                return total_loss, nll_loss, extractive_loss
                # l_probs = F.log_softmax(sentence_prediction, dim=1)
                # extractive_loss = F.nll_loss(l_probs, summarization_targets, ignore_index=1000) # hard code for now
                # total_loss = extractive_loss + loss
                # print(total_loss, nll_loss, extractive_loss)
                # return total_loss, nll_loss, extractive_loss
            elif getattr(model.args, 'sentence_regression', False):
                sentence_prediction = net_output[1]['sentence_prediction']
                sentence_prediction = sentence_prediction.transpose(1, 2)
                span_regression_final = sample['span_regression_final']
                regression_loss = F.mse_loss(sentence_prediction, span_regression_final)
                total_loss = loss + span_lambda * regression_loss
                return total_loss, nll_loss, regression_loss

            
            

        # 3) Previously when loss which didn't take into account sample size:
        #  remember to modify sample_size too when you uncomment this
        # loss_fct = CrossEntropyLoss(ignore_index=self.padding_idx)
        # start_positions = sample['span_start_ids']
        # end_positions = sample['span_end_ids']
        # start_logits = net_output[1]['start_logits']
        # end_logits = net_output[1]['end_logits']
        # start_loss = loss_fct(start_logits, start_positions)
        # end_loss = loss_fct(end_logits, end_positions)
        # total_span_loss = (start_loss + end_loss) / 2   
        # total_loss = loss + total_span_loss
        # if torch.isnan(total_span_loss):
        #     from fairseq import pdb;
        #     pdb.set_trace()
        # return total_loss, total_loss, total_loss

        # 4) Previously when I had a single span prediction and also didn't modify sample_size
        # start_positions = sample['span_ids'][:, 0]
        # end_positions = sample['span_ids'][:, 1]
        # start_logits = net_output[1]["start_logits"].squeeze(-1)
        # end_logits = net_output[1]["end_logits"].squeeze(-1)
        # start_loss = loss_fct(start_logits, start_positions)
        # end_loss = loss_fct(end_logits, end_positions)
        # total_span_loss = (start_loss + end_loss) / 2    
        # total_loss = loss + total_span_loss
        # total_loss = total_span_loss
        # total_loss = total_span_loss
        # if torch.isnan(total_span_loss):
        #     from fairseq import pdb;
        #     pdb.set_trace()
        # # print(total_loss)
        # return total_loss, nll_loss, total_span_loss


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.)
        smooth_loss.masked_fill_(pad_mask, 0.)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion('span_loss')
class SpanCriterion(FairseqCriterion):

    def __init__(self, task, sentence_avg, label_smoothing):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        # self.loss_fct_linear = CrossEntropyLoss(ignore_index=self.padding_idx)

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

        Alex - just added loss_index argument for consistency, but it doesn't do anything here. 
        """
        net_output = model(**sample['net_input'])
        loss, nll_loss, span_loss = self.compute_loss(model, net_output, sample, reduce=reduce, loss_index=loss_index, nll_only=nll_only, span_only=span_only)

        if getattr(model.args, 'sample_size_one', False):
            sample_size = 1
        else:
            sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']

        if isinstance(loss, float):
            loss_return = 0.0
        else:
            loss_return = loss.data
        if isinstance(nll_loss, float):
            nll_return = 0.0
        else:
            nll_return = nll_loss.data
        if isinstance(span_loss, float):
            span_return = 0.0
        else:
            span_return = span_loss.data
        logging_output = {
            'loss': loss_return,
            'nll_loss': nll_return,
            'span_loss': span_return,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True, loss_index=0, nll_only=False, span_only=False):
        return get_span_and_nll_loss(model, net_output, sample, self.eps, self.padding_idx, reduce=True, nll_only=False, span_only=False)

    @staticmethod
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
