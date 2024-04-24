export CUDA_VISIBLE_DEVICES=0,1,2,3

train_ntcir_data() {
    torchrun --nproc_per_node 4 \
    -m FlagEmbedding.reranker.run \
    --output_dir "./outputs/ntcir$1/cds_$2_$5_reranker/lr_$3_bs_$4_epoch_$6" \
    --model_name_or_path "/path/to/model" \
    --train_data "./dat/ntcir$1/data/cds_$2_$5_split_train_reranker_minedHN.jsonl" \
    --learning_rate $3 \
    --fp16 \
    --num_train_epochs $6 \
    --per_device_train_batch_size $4 \
    --gradient_accumulation_steps 4 \
    --dataloader_drop_last True \
    --train_group_size 4 \
    --max_len 512 \
    --weight_decay 0.01
}

for fold in 15
do
    for diameter in 4
    do
        for edge in 5
        do
            for bs in 2
            do
                for lr in 1e-6 1e-5 3e-5 5e-5
                do
                    for epoch in 3 5 10
                    do
                        train_ntcir_data $fold $diameter $lr $bs $edge $epoch
                    done
                done
            done
        done
    done
done