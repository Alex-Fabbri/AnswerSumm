import json


for split in ["train", "val", "test"]:
    with open(f"data/{split}.jsonl") as f, open(f"data/{split}.source", "w") as outs, open(f"data/{split}.target", "w") as outt:
        for line in f:
            data = json.loads(line)
            data = data['data'][0]

            target = data['final_summary']

            answers = data['answers']
            answer_texts = []
            for answer in answers:
                sents = " <S> ".join([x['text'] for x in answer['sents']])
                answer_texts.append(sents)
            answer_strs = " <A> ".join(answer_texts)
            question = data['question'].replace("\n", " ")
            question = ":".join(question.split(":")[1:])

            source = f"{question} <Q> {answer_strs}"
            source = source.replace("\n", " ")
            target = target.replace("\n", " ")

            outs.write(source + "\n")
            outt.write(target + "\n")

        if split == "train":
            with open("data/aug.source") as fs, open("data/aug.target") as ft:
                for lines, linet in zip(fs, ft):
                    source = source.replace("\n", " ")
                    target = linet.replace("\n", " ")

                    outs.write(source + "\n")
                    outt.write(target + "\n")
