# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
BART: Denoising Sequence-to-Sequence Pre-training for
Natural Language Generation, Translation, and Comprehension
"""

import logging

import torch
import torch.nn as nn

from fairseq import utils
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.fairseq_encoder import EncoderOut

from fairseq.models.transformer import TransformerModel, Linear
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.modules.multihead_attention import MultiheadAttention

from .hub_interface import BARTHubInterface
# from fairseq.data.encoders.gpt2_bpe import get_encoder
from fairseq.data import encoders


logger = logging.getLogger(__name__)


@register_model('bart')
class BARTModel(TransformerModel):

    @classmethod
    def hub_models(cls):
        return {
            'bart.base': 'http://dl.fbaipublicfiles.com/fairseq/models/bart.base.tar.gz',
            'bart.large': 'http://dl.fbaipublicfiles.com/fairseq/models/bart.large.tar.gz',
            'bart.large.mnli': 'http://dl.fbaipublicfiles.com/fairseq/models/bart.large.mnli.tar.gz',
            'bart.large.cnn': 'http://dl.fbaipublicfiles.com/fairseq/models/bart.large.cnn.tar.gz',
            'bart.large.xsum': 'http://dl.fbaipublicfiles.com/fairseq/models/bart.large.xsum.tar.gz',
        }

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

        # We follow BERT's random weight initialization
        self.apply(init_bert_params)

        self.classification_heads = nn.ModuleDict()

        # span heads 
        self.span_heads = nn.ModuleDict()
        quant_noise = getattr(args, "quant_noise_pq", 0)
        quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)
        # if getattr(args, "use_multihead", False):
        #     self.attn_start = MultiheadAttention(
        #         args.decoder_embed_dim,
        #         args.decoder_attention_heads,
        #         kdim=getattr(args, "encoder_embed_dim", None),
        #         vdim=getattr(args, "encoder_embed_dim", None),
        #         dropout=args.attention_dropout,
        #         encoder_decoder_attention=True,
        #         q_noise=quant_noise,
        #         qn_block_size=quant_noise_block_size,
        #     )
        #     self.attn_end = MultiheadAttention(
        #         args.decoder_embed_dim,
        #         args.decoder_attention_heads,
        #         kdim=getattr(args, "encoder_embed_dim", None),
        #         vdim=getattr(args, "encoder_embed_dim", None),
        #         dropout=args.attention_dropout,
        #         encoder_decoder_attention=True,
        #         q_noise=quant_noise,
        #         qn_block_size=quant_noise_block_size,
        #     )

        if getattr(self.args, 'sentence_prediction', False):
            if getattr(self.args, 'sentence_prediction_binary', False):
                self.mhead_sentence = MultiheadAttention(
                    args.encoder_embed_dim,
                    args.encoder_attention_heads,
                    kdim=getattr(args, "decoder_embed_dim", None),
                    vdim=getattr(args, "decoder_embed_dim", None),
                    dropout=args.attention_dropout,
                    encoder_decoder_attention=True,
                    q_noise=quant_noise,
                    qn_block_size=quant_noise_block_size,
                )
                self.linear_sentence = Linear(self.mhead_sentence.num_heads, 2)
            elif getattr(self.args, 'sentence_prediction_span_simple_inner', False):
                pass
            elif getattr(self.args, 'sentence_prediction_span_head', False) or \
                getattr(self.args, 'sentence_prediction_span_head_attn', False) or \
                getattr(self.args, 'sentence_prediction_span_head_decoder_query', False):
                self.mhead_sentence = MultiheadAttention(
                    args.encoder_embed_dim,
                    args.encoder_attention_heads,
                    kdim=getattr(args, "decoder_embed_dim", None),
                    vdim=getattr(args, "decoder_embed_dim", None),
                    dropout=args.attention_dropout,
                    encoder_decoder_attention=True,
                    q_noise=quant_noise,
                    qn_block_size=quant_noise_block_size,
                )
                if not getattr(self.args, 'sentence_prediction_span_head_attn', False):
                    self.linear_sentence = Linear(self.mhead_sentence.num_heads, 1)
            elif getattr(self.args, 'sentence_regression', False):
                self.linear_sentence = Linear(1, 1)
        if getattr(self.args, "sentence_prediction_pool_target_eos", False):
            self.pool_target = torch.nn.Conv1d(args.decoder_embed_dim, args.decoder_embed_dim, 2, stride=1)
        if getattr(self.args, "sentence_prediction_pool_source_eos", False):
            self.pool_source = torch.nn.Conv1d(args.encoder_embed_dim, args.encoder_embed_dim, 2, stride=2)

        if getattr(self.args, "nli_reinforce", False):
            self.bpe = encoders.build_bpe(args)
            self.roberta = torch.hub.load('pytorch/fairseq:main', 'roberta.large.mnli')
            #sd = torch.load("/export/home/answersumm/fairseq/roberta.large.mnli/model.pt")['model']
            #self.roberta.load_state_dict(sd)

            for param in self.roberta.parameters():
                param.requires_grad = False
        # if getattr(self.args, "sentence_prediction_pool_target_eos")
        # self.bpe = encoders.build_bpe(args)
        # linear_dict = {"span_heads": nn.Linear(self.args.encoder_embed_dim, 2)}
        # self.span_heads = nn.ModuleDict(linear_dict)

    @staticmethod
    def add_args(parser):
        super(BARTModel, BARTModel).add_args(parser)
        parser.add_argument(
            '--pooler-dropout', type=float, metavar='D',
            help='dropout probability in the masked_lm pooler layers'
        )
        parser.add_argument(
            '--pooler-activation-fn',
            choices=utils.get_available_activation_fns(),
            help='activation function to use for pooler layer'
        )
        parser.add_argument('--use-span', action='store_true',
                            help='whether to use span model')
        parser.add_argument('--dummy-classifier', action='store_true',
                            help='whether to use dummy classifier (for DEBUGGING)')
        parser.add_argument('--dummy-classifier-source', action='store_true',
                            help='whether to use dummy classifier with features from source (for DEBUGGING)')
        parser.add_argument('--dummy-classifier-target', action='store_true',
                            help='whether to use dummy classifier with features from target (for DEBUGGING)')
        parser.add_argument('--decoder-classifier', action='store_true',
                            help='whether to use decoder classifier with features from target')
        parser.add_argument('--sample-size-one', action='store_true',
                            help='whether to use sample size of 1')
        parser.add_argument('--use-encoder-features', action='store_true',
                            help='whether to use encoder features or use another decoder pass')
        parser.add_argument('--use-multihead', action='store_true',
                            help='whether to use multihead decoder over encoder features')
        parser.add_argument('--sentence-prediction', action='store_true',
                            help='whether we are predicting sentences or not')
        parser.add_argument('--sentence-prediction-pool-target-eos', action='store_true',
                            help='whether we are combining bos and eos token representations in the target')
        parser.add_argument('--sentence-prediction-pool-source-eos', action='store_true',
                            help='whether we are combining bos and eos token representations in the source')
        parser.add_argument('--sentence-prediction-before-softmax', action='store_true',
                            help='whether we are getting features from model before applying softmax in mhead attention')
        parser.add_argument('--sentence-prediction-binary', action='store_true',
                            help='whether we are getting features from model before applying softmax in mhead attention')
        parser.add_argument('--sentence-prediction-attn-direct', action='store_true',
                            help='whether we using the cross attention directly from the decoder')
        parser.add_argument('--sentence-prediction-single-span', action='store_true',
                            help='whether predicting a single span which corresponds to an EOS token')
        parser.add_argument('--sentence-prediction-span-simple-inner', action='store_true',
                            help='whether predicting a single span which corresponds to an EOS token')
        parser.add_argument('--sentence-prediction-span-head', action='store_true',
                            help='whether predicting a single span which corresponds to an EOS token')
        parser.add_argument('--sentence-prediction-span-head-attn', action='store_true',
                            help='whether just using span head attn for loss calculations')
        parser.add_argument('--sentence-prediction-span-head-decoder-query', action='store_true',
                            help='whether just using span head attn for loss calculations')
        parser.add_argument('--sentence-regression', action='store_true',
                            help='whether just using span head attn for loss calculations')
        parser.add_argument('--simple-linear', action='store_true',
                            help='whether to use simple linear layer on top of either decoder or encoder features')
        parser.add_argument(
            '--span-loss-lambda', type=float, metavar='L',
            help='what lambda to use in front of span loss'
        )
        parser.add_argument(
            '--nll-loss-lambda', type=float, metavar='L',
            help='what lambda to use in front of NLL loss'
        )
        parser.add_argument('--span-head-complex', action='store_true',
                            help='whether to use more complicated span head')
        parser.add_argument('--sentence-prediction-span-added-linear', action='store_true',
                            help='whether to use additional linear layers before sentence span prediction')
        parser.add_argument('--every-timestep', action='store_true',
                            help='whether we are predicting spans/doing regression on every timestep')
        parser.add_argument('--alternate', action='store_true',
                            help='alternate losses')
        parser.add_argument('--alternate-span-only', action='store_true',
                            help='use only the span loss (and not span + nll) when alternating losses')
        parser.add_argument('--validate-nll', action='store_true',
                            help='use only nll for validation checkpoint')
        parser.add_argument('--validate-nll-pdb', action='store_true',
                            help='debug - use only nll for validation checkpoint')
        parser.add_argument('--nli-reinforce', action='store_true',
                            help='whether to use nli reinforce as the loss function')
        parser.add_argument('--nli-reinforce-only', action='store_true',
                            help='whether to use nli reinforce as the only loss function')
        parser.add_argument('--nli-reinforce-lambda', type=float, metavar='NL',help='what lambda to use in front of nli reinforce loss')
        parser.add_argument('--nli-reinforce-nll-lambda', type=float, metavar='NL',help='what lambda to use in front of nll loss')
        parser.add_argument('--nli-pdb', action='store_true',
                            help='whether to do nli pdb')
        parser.add_argument('--nli-reinforce-nli-only', action='store_true',
                            help='whether to do nli only for reward (overrides rouge)')
        parser.add_argument('--nli-reinforce-rouge-only', action='store_true',
                            help='whether to do rouge only for reward')
        parser.add_argument('--nli-reinforce-normalize-sent', action='store_true',
                            help='whether to normalization on the sentence level')
        parser.add_argument('--nli-reinforce-no-baseline', action='store_true',
                            help='whether to use the baseline in the calulcations or just sampling')
        parser.add_argument('--nli-reinforce-sentence-level', action='store_true',
                            help='nli reinforce option')
        parser.add_argument('--nli-reinforce-selfbleu-only', action='store_true',
                            help='nli reinforce option')
        parser.add_argument('--nli-reinforce-volume-only', action='store_true',
                            help='nli reinforce option')
        parser.add_argument('--nli-reinforce-all', action='store_true',
                            help='nli reinforce option')
        parser.add_argument('--nli-reinforce-nli-rouge-volume', action='store_true',
                            help='nli reinforce option')
        parser.add_argument('--nli-reinforce-nli-rouge-diversity', action='store_true',
                            help='nli reinforce option')
        parser.add_argument('--nli-reinforce-nli-diversity-volume', action='store_true',
                            help='nli reinforce option')
        parser.add_argument('--nli-reinforce-rouge-diversity-volume', action='store_true',
                            help='nli reinforce option')
        parser.add_argument('--nli-reinforce-nli-rouge', action='store_true',
                            help='nli reinforce option')
        parser.add_argument('--nli-reinforce-nli-diversity', action='store_true',
                            help='nli reinforce option')
        parser.add_argument('--nli-reinforce-nli-volume', action='store_true',
                            help='nli reinforce option')
        parser.add_argument('--nli-reinforce-rouge-diversity', action='store_true',
                            help='nli reinforce option')
        parser.add_argument('--nli-reinforce-rouge-volume', action='store_true',
                            help='nli reinforce option')
        parser.add_argument('--nli-reinforce-volume-diversity', action='store_true',
                            help='nli reinforce option')
        parser.add_argument('--volume-no-sent-avg', action='store_true',
                            help='nli reinforce option')
        parser.add_argument('--rotating-nll-rl', action='store_true',
                            help='nli reinforce option')
        parser.add_argument('--nli-reinforce-rotating-rl-only', action='store_true',
                            help='nli reinforce option')

        # parser.add_argument('--encoder-json', help='encoder json for bpe')
        # parser.add_argument('--vocab-bpe', help='vocab bpe')

    @property
    def supported_targets(self):
        return {'self'}

    def forward(
        self, src_tokens, src_lengths, prev_output_tokens,
        prev_input_tokens=None, target_sent_ids=None, summarization_ids_out=None, features_only=False, 
        classification_head_name=None, **kwargs
    ):
        if classification_head_name is not None:
            features_only = True
        # STEP A) - encoder the source
        encoder_out = self.encoder(
            src_tokens,
            src_lengths=src_lengths,
            **kwargs,
        )
        # # STEP B) decoder and either put features in extra or not (default)
        if not getattr(self.args, 'use_span', False):
            x, extra = self.decoder(
                prev_output_tokens,
                encoder_out=encoder_out,
                features_only=features_only,
                features_in_extra=False,
                **kwargs,
            )        
        else:
            if getattr(self.args, 'dummy_classifier_target', False) or \
                getattr(self.args, 'decoder_classifier', False):
                features_in_extra = True
            else:
                features_in_extra = False
            x, extra = self.decoder(
                prev_output_tokens,
                encoder_out=encoder_out,
                features_only=features_only,
                features_in_extra=features_in_extra,
                **kwargs,
            )
            if getattr(self.args, 'dummy_classifier', False):
                if getattr(self.args, 'dummy_classifier_source', False):
                    source_features_orig = encoder_out.encoder_out.transpose(0,1) # this works 
                    sentence_representation = source_features_orig[src_tokens.eq(self.encoder.dictionary.eos()), :].view(source_features_orig.size(0), -1, source_features_orig.size(-1))[:, -1, :]
                
                elif getattr(self.args, 'dummy_classifier_target', False):
                    source_features_orig = extra["features"]
                    sentence_representation = source_features_orig[prev_output_tokens.eq(self.encoder.dictionary.eos()), :].view(source_features_orig.size(0), -1, source_features_orig.size(-1))[:, -1, :]
                else:
                    # dummy classifier full
                    source_features_orig, _ = self.decoder(
                        prev_input_tokens,
                        encoder_out=encoder_out,
                        features_only=True,
                        features_in_extra=False,
                        **kwargs,
                    )
                    sentence_representation = source_features_orig[prev_input_tokens.eq(self.encoder.dictionary.eos()), :].view(source_features_orig.size(0), -1, source_features_orig.size(-1))[:, -1, :]
                classifier_logits = self.span_heads["classifier"](sentence_representation)
                extra['classifier_logits'] = classifier_logits

            elif getattr(self.args, 'simple_linear', False):
                #import pdb;pdb.set_trace()
                if getattr(self.args, 'use_encoder_features', False):
                    source_features_orig = encoder_out.encoder_out.transpose(0,1)
                else:
                    # cur_attn = extra['attn'][0]
                    # encoder_features_final = torch.bmm(cur_attn, encoder_features) # [2, 30, 1024]
                    ##STEP C) run decoder over source again (with it moved to the right to match standard decoder input)
                    source_features_orig, _ = self.decoder(
                            prev_input_tokens,
                            encoder_out=encoder_out,
                            features_only=True,
                            features_in_extra=False,
                            **kwargs,
                        )
                # TODO need to modify the spans too
                # STEP D)
                # select features from target corresponding to <S> tokens. 
                if getattr(self.args, 'every_timestep', False):
                    target_features_selected = extra["features"] # (batch x decoder_len x 1024)
                else:
                    target_features = extra["features"] # (batch x decoder_len x 1024)
                    target_sent_ids = target_sent_ids.unsqueeze(-1)
                    target_sent_ids_mask = (target_sent_ids == -43)
                    target_sent_ids[target_sent_ids==-43] = 1

                    target_sent_ids = target_sent_ids.repeat(1, 1, 1024) # (batch x max_num of spans x 1024)
                    target_sent_ids_mask = target_sent_ids_mask.repeat(1, 1, 1024)
                    target_features_selected = torch.gather(target_features, 1, target_sent_ids)#.clone() # (batch x max_num of spans x 1024)
                    target_features_selected = target_features_selected.masked_fill(target_sent_ids_mask, 1e-8)

                    if getattr(self.args, 'sentence_prediction_pool_target_eos', False):
                        target_features_selected_tmp = target_features_selected_tmp.transpose(1, 2)
                        target_features_selected = self.pool_target(target_features_selected_tmp).transpose(1, 2)
                    else:
                        target_features_selected = target_features_selected


                # STEP E)
                # Expand source features and target features to 2*hidden_dim so that we can get start and end logits
                # x_source is of size  (batch x max source length x 1024). Pass to linear layer
                source_features_linear = self.span_heads["x_source_linear"](source_features_orig) # (batch x max source length x 2048)
                target_features_selected = self.span_heads["x_decoded_linear"](target_features_selected) # (batch x max_num of spans x 2048)
                
                # STEP F)
                source_features_start, source_features_end = source_features_linear.split(1024, dim=-1) # each of size (batch x max_num of spans x 1024)
                target_features_selected_start, target_features_selected_end = target_features_selected.split(1024, dim=-1) # each of size (batch x max_num of spans x 1024)

                # STEP G)
                # Swap the last two dimensions of the target features to make the inner product work below
                target_features_selected_start = target_features_selected_start.transpose(2, 1) #  [batch_size, hidden_dim, max_num_spans]
                target_features_selected_end = target_features_selected_end.transpose(2, 1) #  [batch_size, hidden_dim, max_num_spans]
                
                # STEP H)
                # calculate inner product of source and target features
                start_logits = torch.bmm(source_features_start, target_features_selected_start) # [batch_size, max_source_len, max_num_spans]
                end_logits = torch.bmm(source_features_end, target_features_selected_end) # [batch_size, max_source_len, max_num_spans]

                extra['start_logits'] = start_logits
                extra['end_logits'] = end_logits
                # #when I was just predicting a single span
                # span_out = self.span_heads["span_heads"](x_source) 
                # start_logits, end_logits = span_out.split(1, dim=-1)
            elif getattr(self.args, 'use_multihead', False):
                target_features = extra["features"] # (batch x decoder_len x 1024)
                if getattr(self.args, 'every_timestep', False):
                    target_features_selected = extra["features"] # (batch x decoder_len x 1024)
                else:
                    target_sent_ids = target_sent_ids.unsqueeze(-1)
                    target_sent_ids_expanded = target_sent_ids.repeat(1, 1, 1024) # (batch x max_num of spans x 1024)
                    target_features_selected = torch.gather(target_features, 1, target_sent_ids_expanded)#.clone() # (batch x max_num of spans x 1024)
                    target_features_selected = target_features_selected.transpose(0, 1)
                features = target_features_selected
                
                # features = extra['features'].transpose(0,1)
                tgt_len, bsz, embed_dim = features.size()
                x_dec, attn = self.attn_start(
                    query=features,
                    key=encoder_out.encoder_out ,
                    value=encoder_out.encoder_out ,
                    key_padding_mask=encoder_out.encoder_padding_mask,
                    incremental_state=None,
                    static_kv=True,
                    need_weights=True,
                    before_softmax=True
                )
                # x_dec = (max_num_of_spans x batch x 1024) (e.g. [6, 1, 1024])
                # attn = (batch x max_num of spans x source_len) (e.g. [1, 6, 581])
                # decoder_padding_mask = prev_output_tokens.eq(self.decoder.padding_idx)
                x_dec_enc_q, attn_enc_q = self.attn_start(
                    query=encoder_out.encoder_out ,
                    key=features,
                    value=features,
                    key_padding_mask=None,
                    incremental_state=None,
                    static_kv=True,
                    need_weights=True,
                )
                # attn_enc_q.shape - (batch x source_len x max_num of spans) - torch.Size([1, 581, 6])
                # x_dec_enc_q.shape - (source_len x batch * 1024) - torch.Size([581, 1, 1024])
            elif getattr(self.args, 'sentence_prediction', False):
                # sentence_prediction_attn_direct
                # sentence_prediction_attn_direct
                if not getattr(self.args, 'sentence_prediction_attn_direct', False):
                    pass
                    target_features = extra["features"] # (batch x decoder_len x 1024)

                    if getattr(self.args, 'every_timestep', False):
                        target_features_selected = extra["features"] # (batch x decoder_len x 1024)
                    else:
                        target_sent_ids = target_sent_ids.unsqueeze(-1)
                        target_sent_ids_mask = (target_sent_ids == -43)
                        target_sent_ids[target_sent_ids==-43] = 1

                        target_sent_ids_expanded = target_sent_ids.repeat(1, 1, 1024) # (batch x max_num of spans x 1024)
                        target_sent_ids_mask = target_sent_ids_mask.repeat(1, 1, 1024)
                        target_features_selected_tmp = torch.gather(target_features, 1, target_sent_ids_expanded)#.clone() # (batch x max_num of spans x 1024)
                        target_features_selected_tmp = target_features_selected_tmp.masked_fill(target_sent_ids_mask, 1e-8)
                        if getattr(self.args, 'sentence_prediction_pool_target_eos', False):
                            target_features_selected_tmp = target_features_selected_tmp.transpose(1, 2)
                            target_features_selected = self.pool_target(target_features_selected_tmp).transpose(1, 2)
                        else:
                            target_features_selected = target_features_selected_tmp 

                        
                    encoder_output = encoder_out.encoder_out.transpose(0, 1)
                    # mask the sentence features which correspond to padding
                    sentence_ids = summarization_ids_out.unsqueeze(-1)
                    summarization_ids_out_mask = (sentence_ids == -43)
                    sentence_ids[sentence_ids==-43] = 1
                    sentence_ids = sentence_ids.repeat(1, 1, encoder_output.shape[-1]) # (batch x max_num of sents x 1024)
                    summarization_ids_out_mask = summarization_ids_out_mask.repeat(1, 1, encoder_output.shape[-1])
                    sentence_features_tmp = torch.gather(encoder_output, 1, sentence_ids)#.clone() # (batch x max_num of sents x 1024)
                    sentence_features_tmp = sentence_features_tmp.masked_fill(summarization_ids_out_mask, 1e-8)
                    
                    if getattr(self.args, 'sentence_prediction_span_added_linear', False):
                        sentence_features_tmp = self.span_heads["x_source_linear"](sentence_features_tmp) 
                        target_features_selected = self.span_heads["x_decoded_linear"](target_features_selected) 
                    if getattr(self.args, 'sentence_prediction_pool_source_eos', False):
                        sentence_features_tmp = sentence_features_tmp.transpose(1, 2)
                        sentence_features = self.pool_source(sentence_features_tmp).permute(2, 0, 1)
                    else:
                        sentence_features = sentence_features_tmp.transpose(0, 1)
                if getattr(self.args, 'sentence_prediction_before_softmax', False):
                    if getattr(self.args, 'sentence_prediction_binary', False):
                        sent_enc_dec, _ = self.mhead_sentence(query=sentence_features ,key=target_features_selected,value=target_features_selected,key_padding_mask=None,incremental_state=None,static_kv=True,need_weights=True, before_softmax=True)
                        tgt_len, bsz, embed_dim = sentence_features.size()
                        sent_enc_dec_out = sent_enc_dec.view(bsz, self.mhead_sentence.num_heads, tgt_len, -1).permute(0, 3, 2, 1) # ([2, 1, 27, 16]) = batch_size x max_spans x source_len x num_heads
                        sentence_prediction = self.linear_sentence(sent_enc_dec_out)
                        sentence_prediction = sentence_prediction.permute(0, 3, 1, 2)
                        extra['sentence_prediction'] = sentence_prediction
                    elif getattr(self.args, 'sentence_prediction_span_simple_inner', False):
                        sentence_features = sentence_features.transpose(0,1)
                        target_features_selected = target_features_selected.transpose(1,2)
                        sentence_prediction = torch.bmm(sentence_features, target_features_selected)
                        extra['sentence_prediction'] = sentence_prediction
                    elif getattr(self.args, 'sentence_prediction_span_head', False):
                        sent_enc_dec, _ = self.mhead_sentence(query=sentence_features ,key=target_features_selected,value=target_features_selected,key_padding_mask=None,incremental_state=None,static_kv=True,need_weights=True, before_softmax=True)
                        tgt_len, bsz, embed_dim = sentence_features.size()
                        sent_enc_dec_out = sent_enc_dec.view(bsz, self.mhead_sentence.num_heads, tgt_len, -1).permute(0, 3, 2, 1) # ([2, 1, 27, 16]) = batch_size x max_spans x source_len x num_heads
                        sentence_prediction = self.linear_sentence(sent_enc_dec_out).squeeze(-1)
                        sentence_prediction = sentence_prediction.permute(0, 2, 1)
                        extra['sentence_prediction'] = sentence_prediction
                    elif getattr(self.args, 'sentence_prediction_span_head_attn', False):
                        target_features_selected = target_features_selected.transpose(0, 1)
                        sentence_features = sentence_features.transpose(0,1)
                        _, attn_ = self.mhead_sentence(query=target_features_selected,key=sentence_features,value=sentence_features,key_padding_mask=None,incremental_state=None,static_kv=True,need_weights=True, before_softmax=False)
                        extra['attn_'] = attn_
                    
                    elif getattr(self.args, 'sentence_regression', False):
                        sentence_features = sentence_features.transpose(0,1)
                        target_features_selected = target_features_selected.transpose(1,2)
                        sentence_prediction = torch.bmm(sentence_features, target_features_selected)
                        sentence_prediction = sentence_prediction.unsqueeze(-1)
                        sentence_prediction = self.linear_sentence(sentence_prediction)
                        sentence_prediction = sentence_prediction.squeeze(-1)
                        extra['sentence_prediction'] = sentence_prediction
                        # sent_enc_dec, _ = self.mhead_sentence(query=sentence_features ,key=target_features_selected,value=target_features_selected,key_padding_mask=None,incremental_state=None,static_kv=True,need_weights=True, before_softmax=True)
                        # tgt_len, bsz, embed_dim = sentence_features.size()
                        # sent_enc_dec_out = sent_enc_dec.view(bsz, self.mhead_sentence.num_heads, tgt_len, -1).permute(0, 3, 2, 1) # ([2, 1, 27, 16]) = batch_size x max_spans x source_len x num_heads
                        # sentence_prediction = self.linear_sentence(sent_enc_dec_out)
                        # sentence_prediction = sentence_prediction.permute(0, 3, 1, 2)
                        # extra['sentence_prediction'] = sentence_prediction

                    # elif getattr(self.args, 'sentence_prediction_span_head_decoder_query', False):
                    #     from fairseq import pdb;pdb.set_trace()
                    #     target_features_selected = target_features_selected.transpose(0, 1)
                    #     sentence_features = sentence_features.transpose(0,1)
                    #     sent_enc_dec, _ = self.mhead_sentence(query=target_features_selected,key=sentence_features,value=sentence_features,key_padding_mask=None,incremental_state=None,static_kv=True,need_weights=True, before_softmax=False)
                        
                    #     tgt_len, bsz, embed_dim = target_features_selected.size()
                    #     sent_enc_dec_out = sent_enc_dec.view(bsz, self.mhead_sentence.num_heads, tgt_len, -1).permute(0, 3, 2, 1)
                    #     sentence_prediction = self.linear_sentence(sent_enc_dec_out).squeeze(-1)
                    #     sentence_prediction = sentence_prediction.permute(0, 2, 1)
                    #     extra['sentence_prediction'] = sentence_prediction
                elif getattr(self.args, 'sentence_prediction_attn_direct', False):
                    # target_features = extra["features"] # (batch x decoder_len x 1024)
                    attn_value = extra['attn'][0]

                    target_sent_ids = target_sent_ids.unsqueeze(-1)
                    target_sent_ids_mask = (target_sent_ids == -43)
                    target_sent_ids[target_sent_ids==-43] = 1
                    target_sent_ids_expanded = target_sent_ids.repeat(1, 1, attn_value.shape[-1])
                    target_sent_ids_mask = target_sent_ids_mask.repeat(1, 1, attn_value.shape[-1])
                    attn_value_source = torch.gather(attn_value, 1, target_sent_ids_expanded)
                    attn_value_source = attn_value_source.masked_fill(target_sent_ids_mask, 1e-8)


                    sentence_ids = summarization_ids_out.unsqueeze(-1)
                    summarization_ids_out_mask = (sentence_ids == -43)
                    sentence_ids[sentence_ids==-43] = 1
                    sentence_ids = sentence_ids.transpose(1, 2)
                    summarization_ids_out_mask = summarization_ids_out_mask.transpose(1, 2)
                    sentence_ids_expanded = sentence_ids.repeat(1, attn_value_source.shape[1], 1)
                    summarization_ids_out_mask = summarization_ids_out_mask.repeat(1, attn_value_source.shape[1], 1)
                    attn_value_source_target = torch.gather(attn_value_source, 2, sentence_ids_expanded)
                    attn_value_source_target = attn_value_source_target.masked_fill(summarization_ids_out_mask, 1e-8)

                    extra['attn_out'] = attn_value_source_target

                    
                    # sentence_ids = sentence_ids.repeat(1, 1, encoder_output.shape[-1]) # (batch x max_num of sents x 1024)
                    # summarization_ids_out_mask = summarization_ids_out_mask.repeat(1, 1, encoder_output.shape[-1])
                    # sentence_features_tmp = torch.gather(encoder_output, 1, sentence_ids)#.clone() # (batch x max_num of

                    # target_sent_ids_expanded = target_sent_ids.repeat(1, 1, 1024) # (batch x max_num of spans x 1024)
                    # target_sent_ids_mask = target_sent_ids_mask.repeat(1, 1, 1024)
                    # target_features_selected_tmp = torch.gather(target_features, 1, target_sent_ids_expanded)#.clone() # (batch x max_num of spans x 1024)
                    # target_features_selected_tmp = target_features_selected_tmp.masked_fill(target_sent_ids_mask, 1e-8)
                    print(extra.keys())
                else:
                    raise NotImplementedError()
                    exit()
            # 

                # [2, 16, 27, 1]
                # sent_enc_dec_out = sent_enc_dec.view(bsz, self.mhead_sentence.num_heads, tgt_len, -1).transpose(1, 0)
                # sent_enc_dec_out = sent_enc_dec_out.mean(dim=0)
                # if we just want to use a single prediction rather than 0/1

                # sentence_prediction = self.linear_sentence(sent_enc_dec)
                # sentence_prediction = sentence_prediction.permute(1, 2, 0)
                # extra['sentence_prediction'] = sentence_prediction
                # sent_enc_dec.shape - [27, 2, 1024] (source_sentences x batch_size x 1024)
                # x_dec_enc_q2, attn_enc_q2 = self.attn_start(query=target_features_selected ,key=sentence_features,value=sentence_features,key_padding_mask=None,incremental_state=None,static_kv=True,need_weights=True)

        if classification_head_name is not None:
            sentence_representation = x[
                src_tokens.eq(self.encoder.dictionary.eos()), :
            ].view(x.size(0), -1, x.size(-1))[:, -1, :]
            x = self.classification_heads[classification_head_name](
                sentence_representation
            )
        return x, extra

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        checkpoint_file='model.pt',
        data_name_or_path='.',
        bpe='gpt2',
        **kwargs,
    ):
        from fairseq import hub_utils
        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            bpe=bpe,
            load_checkpoint_heads=True,
            **kwargs,
        )
        return BARTHubInterface(x['args'], x['task'], x['models'][0])

    def register_classification_head(self, name, num_classes=None, inner_dim=None, **kwargs):
        """Register a classification head."""
        logger.info("Registering classification head: {0}".format(name))
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    'and inner_dim {} (prev: {})'.format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        self.classification_heads[name] = BARTClassificationHead(
            self.args.encoder_embed_dim,
            inner_dim or self.args.encoder_embed_dim,
            num_classes,
            self.args.pooler_activation_fn,
            self.args.pooler_dropout,
        )

    def register_span_head(self, name, num_classes=None, inner_dim=None, **kwargs):
        """Register a classification head."""
        logger.info("Registering classification head: {0}".format(name))
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    'and inner_dim {} (prev: {})'.format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        # span-head-complex
        if getattr(self.args, 'span_head_complex', False):
            self.span_heads[name] = BARTSpanHeadComplex(
                self.args.encoder_embed_dim,
                inner_dim or self.args.encoder_embed_dim,
                num_classes,
                self.args.pooler_activation_fn,
                self.args.pooler_dropout,
            )
        elif getattr(self.args, 'sentence_prediction_span_added_linear', False):
            self.span_heads[name] = BARTSpanHead(
                self.args.encoder_embed_dim,
                1024,
            )
        else:
            self.span_heads[name] = BARTSpanHead(
                self.args.encoder_embed_dim,
                num_classes,
            )
            

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        prefix = name + '.' if name != '' else ''
        current_head_names = [] if not hasattr(self, 'classification_heads') else \
            self.classification_heads.keys()
        current_span_names = [] if not hasattr(self, 'span_heads') else \
            self.span_heads.keys()

        # Handle new classification heads present in the state dict.
        keys_to_delete = []
        for k in state_dict.keys():
            if not k.startswith(prefix + 'classification_heads.') or not k.startswith(prefix + 'span_heads.'):
                continue
            try:
                head_name = k[len(prefix + 'classification_heads.'):].split('.')[0]
                num_classes = state_dict[prefix + 'classification_heads.' + head_name + '.out_proj.weight'].size(0)
                inner_dim = state_dict[prefix + 'classification_heads.' + head_name + '.dense.weight'].size(0)

                if getattr(self.args, 'load_checkpoint_heads', False):
                    if head_name not in current_head_names:
                        self.register_classification_head(head_name, num_classes, inner_dim)
                else:
                    if head_name not in current_head_names:
                        logger.warning(
                            'deleting classification head ({}) from checkpoint '
                            'not present in current model: {}'.format(head_name, k)
                        )
                        keys_to_delete.append(k)
                    elif (
                        num_classes != self.classification_heads[head_name].out_proj.out_features
                        or inner_dim != self.classification_heads[head_name].dense.out_features
                    ):
                        logger.warning(
                            'deleting classification head ({}) from checkpoint '
                            'with different dimensions than current model: {}'.format(head_name, k)
                        )
                        keys_to_delete.append(k)
            except:
                head_name = k[len(prefix + 'span_heads.'):].split('.')[0]
                num_classes = state_dict[prefix + 'span_heads.' + head_name + '.out_proj.weight'].size(0)
                inner_dim = state_dict[prefix + 'span_heads.' + head_name + '.dense.weight'].size(0)

                if getattr(self.args, 'load_checkpoint_heads', False):
                    if head_name not in current_span_names:
                        self.register_classification_head(head_name, num_classes, inner_dim)
                else:
                    if head_name not in current_span_names:
                        logger.warning(
                            'deleting span_heads head ({}) from checkpoint '
                            'not present in current model: {}'.format(head_name, k)
                        )
                        keys_to_delete.append(k)
                    elif (
                        num_classes != self.classification_heads[head_name].out_proj.out_features
                        or inner_dim != self.classification_heads[head_name].dense.out_features
                    ):
                        logger.warning(
                            'deleting span_heads head ({}) from checkpoint '
                            'with different dimensions than current model: {}'.format(head_name, k)
                        )
                        keys_to_delete.append(k)
        for k in keys_to_delete:
            del state_dict[k]

        def truncate_emb(key):
            if key in state_dict:
                state_dict[key] = state_dict[key][:-1, :]

        # When finetuning on translation task, remove last row of
        # embedding matrix that corresponds to mask_idx token.
        loaded_dict_size = state_dict['encoder.embed_tokens.weight'].size(0)
        if loaded_dict_size == len(self.encoder.dictionary) + 1 and '<mask>' not in self.encoder.dictionary:
            truncate_emb('encoder.embed_tokens.weight')
            truncate_emb('decoder.embed_tokens.weight')
            truncate_emb('encoder.output_projection.weight')
            truncate_emb('decoder.output_projection.weight')

        # When continued pretraining on new set of languages for mbart,
        # add extra lang embeddings at the end of embed_tokens.
        # Note: newly added languages are assumed to have been added at the end.
        if self.args.task == 'multilingual_denoising' and loaded_dict_size < len(self.encoder.dictionary):
            logger.info(
                "Adding extra language embeddings not found in pretrained model for "\
                "continued pretraining of MBART on new set of languages."
            )
            loaded_mask_token_embedding = state_dict['encoder.embed_tokens.weight'][-1, :]

            num_langids_to_add = len(self.encoder.dictionary) - loaded_dict_size
            embed_dim = state_dict['encoder.embed_tokens.weight'].size(1)

            new_lang_embed_to_add = torch.zeros(num_langids_to_add, embed_dim)
            nn.init.normal_(
                new_lang_embed_to_add,
                mean=0,
                std=embed_dim ** -0.5
            )
            new_lang_embed_to_add = new_lang_embed_to_add.to(
                dtype=state_dict['encoder.embed_tokens.weight'].dtype,
            )

            state_dict['encoder.embed_tokens.weight'] = torch.cat([
                state_dict['encoder.embed_tokens.weight'][:loaded_dict_size-1, :],
                new_lang_embed_to_add,
                loaded_mask_token_embedding.unsqueeze(0)]
            )
            state_dict['decoder.embed_tokens.weight'] = torch.cat([
                state_dict['decoder.embed_tokens.weight'][:loaded_dict_size-1, :],
                new_lang_embed_to_add,
                loaded_mask_token_embedding.unsqueeze(0)]
            )

        # Copy any newly-added classification heads into the state dict
        # with their current weights.
        if hasattr(self, 'classification_heads'):
            cur_state = self.classification_heads.state_dict()
            for k, v in cur_state.items():
                if prefix + 'classification_heads.' + k not in state_dict:
                    # logger.info('Overwriting', prefix + 'classification_heads.' + k)
                    state_dict[prefix + 'classification_heads.' + k] = v

        if hasattr(self, 'span_heads'):
            cur_state = self.span_heads.state_dict()
            for k, v in cur_state.items():
                if prefix + 'span_heads.' + k not in state_dict:
                    # TODO(alex-fabbri) This syntax might have been just throwing the error 
                    # and maybe I didn't need to do all this register head stuff
                    # logger.info('Overwriting', prefix + 'span_heads.' + k)
                    logger.info("We are converting this")
                    state_dict[prefix + 'span_heads.' + k] = v

        if hasattr(self, 'attn_start'):
            cur_state =  self.attn_start.state_dict()
            for k, v in cur_state.items():
                if prefix + 'attn_start.' + k not in state_dict:
                    # TODO(alex-fabbri) This syntax might have been just throwing the error 
                    # and maybe I didn't need to do all this register head stuff
                    # logger.info('Overwriting', prefix + 'span_heads.' + k)
                    logger.info("We are converting this")
                    state_dict[prefix + 'attn_start.' + k] = v
        if hasattr(self, 'attn_end'):
            cur_state =  self.attn_end.state_dict()
            for k, v in cur_state.items():
                if prefix + 'attn_end.' + k not in state_dict:
                    # TODO(alex-fabbri) This syntax might have been just throwing the error 
                    # and maybe I didn't need to do all this register head stuff
                    # logger.info('Overwriting', prefix + 'span_heads.' + k)
                    logger.info("We are converting this")
                    state_dict[prefix + 'attn_end.' + k] = v

        if hasattr(self, 'mhead_sentence'):
            cur_state =  self.mhead_sentence.state_dict()
            for k, v in cur_state.items():
                if prefix + 'mhead_sentence.' + k not in state_dict:
                    # TODO(alex-fabbri) This syntax might have been just throwing the error 
                    # and maybe I didn't need to do all this register head stuff
                    # logger.info('Overwriting', prefix + 'span_heads.' + k)
                    logger.info("We are converting this")
                    state_dict[prefix + 'mhead_sentence.' + k] = v
       
        if hasattr(self, 'linear_sentence'):
            cur_state =  self.linear_sentence.state_dict()
            for k, v in cur_state.items():
                if prefix + 'linear_sentence.' + k not in state_dict:
                    # TODO(alex-fabbri) This syntax might have been just throwing the error 
                    # and maybe I didn't need to do all this register head stuff
                    # logger.info('Overwriting', prefix + 'span_heads.' + k)
                    logger.info("We are converting this")
                    state_dict[prefix + 'linear_sentence.' + k] = v

        if hasattr(self, 'roberta'):
            cur_state =  self.roberta.state_dict()
            for k, v in cur_state.items():
                if prefix + 'roberta.' + k not in state_dict:
                    # TODO(alex-fabbri) This syntax might have been just throwing the error 
                    # and maybe I didn't need to do all this register head stuff
                    # logger.info('Overwriting', prefix + 'span_heads.' + k)
                    logger.info("We are converting this")
                    state_dict[prefix + 'roberta.' + k] = v
        
        if hasattr(self, 'pool_target'):
            cur_state =  self.pool_target.state_dict()
            for k, v in cur_state.items():
                if prefix + 'pool_target.' + k not in state_dict:
                    # TODO(alex-fabbri) This syntax might have been just throwing the error 
                    # and maybe I didn't need to do all this register head stuff
                    # logger.info('Overwriting', prefix + 'span_heads.' + k)
                    logger.info("We are converting this")
                    state_dict[prefix + 'pool_target.' + k] = v

        if hasattr(self, 'pool_source'):
            cur_state =  self.pool_source.state_dict()
            for k, v in cur_state.items():
                if prefix + 'pool_source.' + k not in state_dict:
                    # TODO(alex-fabbri) This syntax might have been just throwing the error 
                    # and maybe I didn't need to do all this register head stuff
                    # logger.info('Overwriting', prefix + 'span_heads.' + k)
                    logger.info("We are converting this")
                    state_dict[prefix + 'pool_source.' + k] = v

class BARTClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        activation_fn,
        pooler_dropout,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, num_classes)
        # self.dense = nn.Linear(input_dim, inner_dim)
        # self.activation_fn = utils.get_activation_fn(activation_fn)
        # self.dropout = nn.Dropout(p=pooler_dropout)
        # self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, features, **kwargs):
        x = features
        # x = self.dropout(x)
        x = self.dense(x)
        # x = self.activation_fn(x)
        # x = self.dropout(x)
        # x = self.out_proj(x)
        return x

class BARTSpanHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        num_classes,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, num_classes)

    def forward(self, features, **kwargs):
        x = features
        x = self.dense(x)
        return x

class BARTSpanHeadComplex(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        activation_fn,
        pooler_dropout,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x




@register_model_architecture('bart', 'bart_large')
def bart_large_architecture(args):
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 4*1024)
    args.encoder_layers = getattr(args, 'encoder_layers', 12)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 16)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', True)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 12)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 16)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', False)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', True)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.max_target_positions = getattr(args, 'max_target_positions', 1024)
    args.max_source_positions = getattr(args, 'max_source_positions', 1024)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', True)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', True)

    args.decoder_output_dim = getattr(args, 'decoder_output_dim', args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim', args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, 'no_scale_embedding', True)
    args.layernorm_embedding = getattr(args, 'layernorm_embedding', True)

    args.activation_fn = getattr(args, 'activation_fn', 'gelu')
    args.pooler_activation_fn = getattr(args, 'pooler_activation_fn', 'tanh')
    args.pooler_dropout = getattr(args, 'pooler_dropout', 0.0)


@register_model_architecture('bart', 'bart_base')
def bart_base_architecture(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 768)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 4*768)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 12)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 12)
    bart_large_architecture(args)


@register_model_architecture('bart', 'mbart_large')
def mbart_large_architecture(args):
    args.no_scale_embedding = getattr(args, 'no_scale_embedding', False)
    bart_large_architecture(args)


@register_model_architecture('bart', 'mbart_base')
def mbart_base_architecture(args):
    args.no_scale_embedding = getattr(args, 'no_scale_embedding', False)
    bart_base_architecture(args)


@register_model_architecture('bart', 'mbart_base_wmt20')
def mbart_base_wmt20_architecture(args):
    args.layernorm_embedding = getattr(args, 'layernorm_embedding', False)
    mbart_base_architecture(args)
