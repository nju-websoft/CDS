
export CUDA_VISIBLE_DEVICES=0,1,2,3

train_ntcir_data() {
    torchrun --nproc_per_node 4 \
    -m FlagEmbedding.baai_general_embedding.finetune.run \
    --output_dir "./outputs/ntcir$1/cds_$2_$5/lr_$3_bs_$4_epoch_$6" \
    --model_name_or_path "/path/to/model" \
    --train_data "./data/ntcir$1/data/cds_$2_$5_split_train_minedHN.jsonl" \
    --learning_rate $3 \
    --fp16 \
    --num_train_epochs $6 \
    --per_device_train_batch_size $4 \
    --dataloader_drop_last True \
    --normlized True \
    --temperature 0.02 \
    --query_max_len 16 \
    --passage_max_len 512 \
    --train_group_size 4 \
    --negatives_cross_device \
    --query_instruction_for_retrieval "Represent this sentence for searching relevant passages: "
}

for version in 15
do
    for diameter in 3 4
    do
        for edge in 5
        do
            for bs in 2
            do
                for lr in 1e-6 1e-5 3e-5 5e-5
                do
                    for epoch in 3 5 10
                    do
                        train_ntcir_data $version $diameter $lr $bs $edge $epoch
                    done
                done
            done
        done
    done
done