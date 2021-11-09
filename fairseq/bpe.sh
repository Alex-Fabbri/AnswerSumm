


wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'


FOLDER=$1
for SPLIT in train val test
do
  for LANG in source target
  do
    python -m examples.roberta.multiprocessing_bpe_encoder \
    --encoder-json encoder.json \
    --vocab-bpe vocab.bpe \
    --inputs "$FOLDER/$SPLIT.$LANG" \
    --outputs "$FOLDER/$SPLIT.bpe.$LANG" \
    --workers 60 \
    --keep-empty;
  done
done

  # --only-source \

INPUT_DIR=$1
OUTPUT_DIR=$1-bin
fairseq-preprocess \
  --source-lang "source" \
  --target-lang "target" \
  --trainpref "$INPUT_DIR/train.bpe" \
  --validpref "$INPUT_DIR/val.bpe" \
  --destdir "$OUTPUT_DIR" \
  --workers 60 \
  --srcdict dict.txt \
  --tgtdict dict.txt;
