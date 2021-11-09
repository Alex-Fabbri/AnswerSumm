import json
import sys
import os


folder = sys.argv[1]
folder += "/"
hypfname = sys.argv[2]
if not os.path.exists(folder + hypfname + "_folder"):
    os.mkdir(folder + hypfname + "_folder")

# path to test.source
source = folder + "/test.source"
nli_input = folder + hypfname + "_folder/" + hypfname + ".nli.sent.input"
nli_meta = folder + hypfname + "_folder/" + hypfname + ".nli.sent.meta"
start = 0
total = 0
total_sents = 0
with open(source) as src, open(folder + hypfname) as hypf, open(nli_input, "w") as outputf, \
    open(nli_meta, "w") as outputmetaf:
    for count, (srcl, hypl) in enumerate(zip(src, hypf)):
        if count % 100 == 0:
            print(count)
        src_answers = srcl.strip().split("<Q>")[-1]
        src_answers = src_answers.replace("<A>", "")
        src_answers = src_answers.split("<S>")
        src_answers = [ans.strip() for ans in src_answers if (not ans.isspace() and len(ans) > 0)]
        src_count = len(src_answers)
        
        tgt_sents = hypl.strip().split("<S>")
        if len(tgt_sents[-1]) == 0 or tgt_sents[-1].isspace() or len(tgt_sents[-1]) == 0:
            tgt_sents = tgt_sents[:-1]
            
        tgt_sents = [sent.strip() for sent in tgt_sents]
        tgt_sents = [x.split("    ") for x in tgt_sents]
        tgt_sents = [item for sublist in tgt_sents for item in sublist]
        tgt_sents = [ans.strip() for ans in tgt_sents if (not ans.isspace() and len(ans) > 0)]

        tgt_count = len(tgt_sents)
        total += src_count * tgt_count
        meta_dict = {}
        meta_dict[count] = {}
        meta_dict[count]["full_answer"] = (start, total)
        start = total
        individual_sents = []
        sent_count = 0
        for i in range(tgt_count):
            individual_sents.append((sent_count, sent_count + src_count))
            sent_count += src_count
        total_sents += len(individual_sents)
        meta_dict[count]['individual_sents'] = individual_sents
        json.dump(meta_dict, outputmetaf)
        outputmetaf.write("\n")
        for tgt_sent in tgt_sents:
            for src_answer in src_answers:
                tgt_sent = tgt_sent.replace("\t", " ")
                src_answer = src_answer.replace("\t", " ").replace("<S>", " ")
                outputf.write(src_answer + "\t" + tgt_sent + "\t" + "0\n")
print(f"cd {folder + hypfname}_folder/")
print(f"CUDA_VISIBLE_DEVICES= python nli_eval_data_single.py {folder + hypfname}_folder/{hypfname}.nli.sent.input")


