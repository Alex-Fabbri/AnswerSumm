import sys
import time
import torch
from fairseq.data.data_utils import collate_tokens
from scipy.special import softmax

# # Download RoBERTa already finetuned for MNLI
roberta = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli')
roberta.eval()  # disable dropout for evaluation
roberta.cuda()

bsz = 128
ex_count = 0
slines = []
out_count = 0



filename = sys.argv[1]
inputf = filename
outputf = filename + ".out"
outputlogsf = filename + ".out.softmax"

with open(inputf) as src, \
    open(outputf, "w") as outputf, \
    open(outputlogsf, "w") as outputf2:
    for counter, line in enumerate(src):
        data = line.split("\t")
        claim = data[1]
        evidence = data[0]
        if ex_count % bsz == 0 and ex_count != 0:
            print(counter)
            with torch.no_grad():
                batch = []
                for pair in slines:
                    enc_0 = roberta.encode(pair[0]).tolist()
                    enc_1 = roberta.encode(pair[1]).tolist()
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
                logprobs = roberta.predict('mnli', batch)
                for prob in logprobs:
                    out = list(softmax(prob.cpu().numpy()))
                    outputf2.write(str(out) + "\n")
                hypotheses_batch = logprobs.argmax(dim=1)
                out_count += len(hypotheses_batch)
            for hypothesis in hypotheses_batch:
                outputf.write(str(hypothesis.item()) + '\n')
                outputf.flush()
            slines = []
        slines.append((evidence, claim))
        ex_count += 1
    if slines != []:
        with torch.no_grad():
            batch = []
            for pair in slines:
                enc_0 = roberta.encode(pair[0]).tolist()
                enc_1 = roberta.encode(pair[1]).tolist()
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
            logprobs = roberta.predict('mnli', batch)
            for prob in logprobs:
                out = list(softmax(prob.cpu().numpy()))
                outputf2.write(str(out) + "\n")
            hypotheses_batch = logprobs.argmax(dim=1)
            for hypothesis in hypotheses_batch:
                outputf.write(str(hypothesis.item()) + '\n')
                outputf.flush()
print(out_count)


