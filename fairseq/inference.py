
import sys
import torch
from fairseq.models.bart import BARTModel

#ckpt_path = sys.argv[1]
#bin_path = sys.argv[2]
#output_path = sys.argv[3]
#output_fname = sys.argv[4]

#ckpt_path = "/export/home/answersumm/fairseq/checkpoints/bart-aug-sure"
ckpt_path = "/export/home/answersumm/fairseq/checkpoints/srini_olddata"
bin_path = "/export/home/answersumm/old_aug_sure-bin"
output_path = "/export/home/answersumm/run_from_srini"
output_fname = "bart-old-aug-olddata.txt"

bart = BARTModel.from_pretrained(
    ckpt_path,
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path=bin_path,
    gpt2_encoder_json='/export/home/answersumm/fairseq/encoder.json', 
    gpt2_vocab_bpe='/export/home/answersumm/fairseq/vocab.bpe'
)

bart.cuda()
bart.eval()
bart.half()
count = 1
bsz = 32
with open(f'{output_path}/test.source') as source, open(f'{output_path}/{output_fname}', 'w') as fout:
    sline = source.readline().strip()
    slines = [sline]
    for count, sline in enumerate(source):
        if count % bsz == 0:
            print(count)
            with torch.no_grad():
                #import pdb;pdb.set_trace()
                hypotheses_batch = bart.sample(slines, beam=4, lenpen=2.0, max_len_b=160, min_len=10, no_repeat_ngram_size=3)
            for hypothesis in hypotheses_batch:
                fout.write(hypothesis + '\n')
                fout.flush()
            slines = []

        slines.append(sline.strip())
        count += 1
    if slines != []:
        with torch.no_grad():
        	hypotheses_batch = bart.sample(slines, beam=4, lenpen=2.0, max_len_b=160, min_len=10, no_repeat_ngram_size=3)
        	for hypothesis in hypotheses_batch:
        	    fout.write(hypothesis + '\n')
        	    fout.flush()
