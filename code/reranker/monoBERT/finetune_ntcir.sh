#! /bin/bash

export  CUDA_VISIBLE_DEVICES=4

train_ntcir_data() {
    python run_monobert_nticr_e.py \
        --seed 42 --epoch_num $6 \
        --model_name /path/to/model \
        --train_path ./data/ntcir$4/data/cds_$1_$5_split_train.json \
        --dev_path ./data/ntcir$4/data/cds_$1_$5_split_dev.json \
        --output_dir ./outputs/ntcir$4 \
        --results_save_path ./results/ntcir$4 \
        --lr $2 \
        --train_batch_size $3 \
        --gradient_accumulation_steps $3 \
        --task cds_$1_$5
}



for version in 15
do
    for diameter in 4 5
    do
        for edge in 5
        do
            for bs in 16
            do
                for lr in 1e-6 1e-5 3e-5 5e-5
                do
                    for epoch in 5 10
                    do
                        train_ntcir_data $diameter $lr $bs $version $edge $epoch
                    done
                done
            done
        done
    done
done