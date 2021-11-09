#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import contextlib
import sys

from collections import Counter
from multiprocessing import Pool

from fairseq.data.encoders.gpt2_bpe import get_encoder

# from https://stackoverflow.com/questions/17870544/find-starting-and-ending-indices-of-sublist-in-list
def find_sub_list(sl,l):
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            return ind, ind+sll-1


def main():
    """
    Helper script to encode raw text with the GPT-2 BPE using multiple processes.

    The encoder.json and vocab.bpe files can be obtained here:
    - https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json
    - https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder-json",
        help='path to encoder.json',
    )
    parser.add_argument(
        "--vocab-bpe",
        type=str,
        help='path to vocab.bpe',
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=['-'],
        help="input files to filter/encode",
    )
    parser.add_argument(
        "--inputs_spans",
        nargs="+",
        default=['-'],
        help="input files to filter/encode with span information",
    )
    parser.add_argument(
        "--outputs",
        nargs="+",
        default=['-'],
        help="path to save encoded outputs",
    )
    parser.add_argument(
        "--outputs_spans",
        nargs="+",
        default=['-'],
        help="path to save encoded outputs",
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="keep empty lines",
    )
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()

    assert len(args.inputs) == len(args.outputs), \
        "number of input and output paths should match"

    with contextlib.ExitStack() as stack:
        inputs = [
            stack.enter_context(open(input, "r", encoding="utf-8"))
            if input != "-" else sys.stdin
            for input in args.inputs
        ]
        # inputs_spans = [
        #     stack.enter_context(open(input_spans, "r", encoding="utf-8"))
        #     if input_spans != "-" else sys.stdin
        #     for input_spans in args.inputs_spans
        # ]
        outputs = [
            stack.enter_context(open(output, "w", encoding="utf-8"))
            if output != "-" else sys.stdout
            for output in args.outputs
        ]
        outputs_spans = [
            stack.enter_context(open(output, "w", encoding="utf-8"))
            if output != "-" else sys.stdout
            for output in args.outputs_spans
        ]

        encoder = MultiprocessingEncoder(args)
        pool = Pool(args.workers, initializer=encoder.initializer)
        # pool = Pool(1, initializer=encoder.initializer)
        encoded_lines = pool.imap(encoder.encode_lines, zip(*inputs), 100)

        stats = Counter()
        for i, (filt, enc_lines, span_lines) in enumerate(encoded_lines, start=1):
            if filt == "PASS":
                for line_count, (enc_line, output_h, output_h_span) in enumerate(zip(enc_lines, outputs, outputs_spans)):
                    if span_lines != []:
                        span_line = span_lines[line_count]
                        print(span_line, file=output_h_span)
                    print(enc_line, file=output_h)
            else:
                stats["num_filtered_" + filt] += 1
            if i % 10000 == 0:
                print("processed {} lines".format(i), file=sys.stderr)

        for k, v in stats.most_common():
            print("[{}] filtered {} lines".format(k, v), file=sys.stderr)


class MultiprocessingEncoder(object):

    def __init__(self, args):
        self.args = args

    def initializer(self):
        global bpe
        bpe = get_encoder(self.args.encoder_json, self.args.vocab_bpe)

    def encode(self, line):
        global bpe
        ids = bpe.encode(line)
        return list(map(str, ids))

    def decode(self, tokens):
        global bpe
        return bpe.decode(tokens)

    ## xsum encode_lines
    # def encode_lines(self, lines):
    #     """
    #     Encode a set of lines. All lines will be encoded together.
    #     """
    #     enc_lines = []
    #     enc_spans = []
    #     for count, line_text in enumerate(lines):
    #         # if "w do I discontinue participation in Yahoo" in line_text:
    #         #     from fairseq import pdb;pdb.set_trace()
    #         line_text = line_text.strip()
    #         if len(line_text) == 0 and not self.args.keep_empty:
    #             return ["EMPTY", None, []]
    #         actual_line = line_text.split("MYSEP")[0]
    #         tokens = self.encode(actual_line)
    #         span_sents = line_text.split("MYSEP")[1].split("SENT_SEP")
    #         tmp_spans = []
    #         for span_sent in span_sents:
    #             span_sent = span_sent.strip()
    #             if " " + span_sent in actual_line:
    #                 span_sent = " " + span_sent
    #             else:
    #                 span_sent = span_sent
    #             if "." + span_sent in actual_line and span_sent[0] != " ":
    #                 tmp = actual_line.split(span_sent)
    #                 final = tmp[0] + " " + span_sent + tmp[1]
    #                 span_sent = " " + span_sent
    #                 tokens = self.encode(final)
    #             if "\"" + span_sent in actual_line and span_sent[0] != " ":
    #                 span_sent = "\"" + span_sent
    #             if "grand master of the Grand Orange Lodge of Ireland, said" in span_sent:
    #                 span_sent += "\"â€ª"
    #             span_ids = self.encode(span_sent)
    #             try:
    #                 start, end = find_sub_list(span_ids, tokens)
    #                 tmp_spans.append(str(start))
    #                 tmp_spans.append(str(end))
    #             except Exception as e:
    #                 try:
    #                     start, end = find_sub_list(span_ids[1:], tokens)
    #                     tmp_spans.append(str(start))
    #                     tmp_spans.append(str(end))
    #                 except:
    #                     try:
    #                         start, end = find_sub_list(span_ids[:-1], tokens)
    #                         tmp_spans.append(str(start))
    #                         tmp_spans.append(str(end))
    #                     except:
    #                         try:
    #                             start, end = find_sub_list(span_ids[1:-1], tokens)
    #                             tmp_spans.append(str(start))
    #                             tmp_spans.append(str(end))
    #                         except:
    #                             print(count)
    #                             from fairseq import pdb;pdb.set_trace()
    #                             print(e)
    #                             # print(tokens, span_ids)
    #                             print(span_sent)
    #                             print(actual_line)
    #                             print("HI")
            
    #         enc_spans.append(" ".join(tmp_spans))
    #         enc_lines.append(" ".join(tokens))
    #     return ["PASS", enc_lines, enc_spans]

    def encode_lines(self, lines):
        """
        Encode a set of lines. All lines will be encoded together.
        """
        bad = 0
        enc_lines = []
        enc_spans = []
        for line_text in lines:
            line_text = line_text.strip()
            if len(line_text) == 0 and not self.args.keep_empty:
                return ["EMPTY", None, []]
            actual_line = line_text.split("MYSEP")[0]
            tokens = self.encode(actual_line)
            span_sents = line_text.split("MYSEP")[1]
            enc_spans.append(span_sents)
            enc_lines.append(" ".join(tokens))
        return ["PASS", enc_lines, enc_spans]

    def decode_lines(self, lines):
        dec_lines = []
        for line in lines:
            tokens = map(int, line.strip().split())
            dec_lines.append(self.decode(tokens))
        return ["PASS", dec_lines]


if __name__ == "__main__":
    main()
