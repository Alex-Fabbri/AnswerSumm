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
    --restore-file $BART_PATH \
    --max-tokens $MAX_TOKENS \
    --task translation \
    --source-lang source --target-lang target \
    --truncate-source \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --arch bart_large \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --fp16 --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --save-interval-updates 1000 \
    --save-dir $CHECKPOINT \
    --tensorboard-logdir $TENSORBOARD_DIR \
    --keep-last-epochs 6 \
    --patience 3 \
    --find-unused-parameters;

