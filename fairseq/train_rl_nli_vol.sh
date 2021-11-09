TOTAL_NUM_UPDATES=20000  
WARMUP_UPDATES=500      
LR=3e-05
MAX_TOKENS=1024
UPDATE_FREQ=16

DATA=
CHECKPOINT=
TENSORBOARD_DIR=
BART_PATH=
ENC_PATH=
VOC_PATH=



MKL_THREADING_LAYER=GNU NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 fairseq-train $DATA \
    --max-tokens $MAX_TOKENS \
    --restore-file $BART_PATH \
    --task translation_span \
    --reset-optimizer --reset-dataloader --reset-meters \
    --source-lang source --target-lang target --span-lang span \
    --truncate-source \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --required-batch-size-multiple 1 \
    --arch bart_large \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --ddp-backend=no_c10d \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --skip-invalid-size-inputs-valid-test \
    --save-dir $CHECKPOINT \
    --log-interval 5 \
    --tensorboard-logdir $TENSORBOARD_DIR \
    --find-unused-parameters \
    --update-freq $UPDATE_FREQ \
    --fp16  \
    --sample-size-one \
    --bpe gpt2 \
    --gpt2-encoder-json $ENC_PATH \
    --gpt2-vocab-bpe $VOC_PATH \
    --nli-reinforce \
    --nli-reinforce-nli-volume \
    --nli-reinforce-nll-lambda 0.1 \
    --nli-reinforce-lambda 0.9 \
    --keep-last-epochs 6 \
    --patience 3 ;

