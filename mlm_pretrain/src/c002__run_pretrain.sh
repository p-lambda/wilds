######################## CivilComments ########################

dt=`date '+%Y%m%d_%H%M%S'`
data_dir="mlm_pretrain/data"
TRAIN_FILE="${data_dir}/civilcomments_train.json"
VAL_FILE="${data_dir}/civilcomments_val.json"
model="distilbert-base-uncased"
outdir="${data_dir}/_run__${model}__civilcomments__b32a256_lr1e-4__${dt}"
mkdir -p $outdir

CUDA_VISIBLE_DEVICES=1 python3.7 -u mlm_pretrain/src/run_mlm.py \
    --model_name_or_path $model \
    --train_file $TRAIN_FILE --validation_file $VAL_FILE \
    --do_train --do_eval --output_dir $outdir --overwrite_output_dir \
    --line_by_line --pad_to_max_length --max_seq_length 300 \
    --fp16 --preprocessing_num_workers 10 --learning_rate 1e-4 \
    --max_steps 2000 --logging_first_step --logging_steps 20 --save_steps 100 \
    --evaluation_strategy steps --eval_steps 100 \
    --per_device_train_batch_size 32 --per_device_eval_batch_size 64 --gradient_accumulation_steps 256 \
    |& tee $outdir/log.txt
