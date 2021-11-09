
import sys
import json
from ast import literal_eval

input_filename = sys.argv[1]
meta_file = input_filename[:-6] + ".meta"
logits_file = input_filename + ".out.softmax"
out_file = input_filename + ".out.final"
support_data = []
with open(logits_file) as inputf:
    for count, line in enumerate(inputf):
        if count % 10000 == 0:
            print(count)
        support = float(literal_eval(line)[-1])
        support_data.append(support)
evidence_scores = []
all_scores = []
total_items = 0
bad = 0
with open(meta_file) as inputf, open(out_file, "w") as outputf:
    for count, line in enumerate(inputf):
        try:
            if count % 100 == 0:
                print(count)
            cur_scores = []
            cur_meta = literal_eval(line)[str(count)]
            sents = cur_meta['individual_sents']
            cur_lines = cur_meta['full_answer']
            cur_support = support_data[cur_lines[0]: cur_lines[1]]
            for sent in sents:
                cur_sent_support = cur_support[sent[0]: sent[1]]
                max_cur_sent_support = max(cur_sent_support)
                cur_scores.append(max_cur_sent_support)
                total_items += len(cur_sent_support)
            outputf.write(str(cur_scores) + "\n")
            try:
                cur_mean = sum(cur_scores)/float(len(cur_scores))
                all_scores.append(cur_mean)
            except:
                bad += 1
                continue
        except:
            continue
total_score = sum(all_scores)/len(all_scores)
print(total_score)
assert len(support_data) == total_items
