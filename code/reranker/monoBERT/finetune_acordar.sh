#! /bin/bash

export  CUDA_VISIBLE_DEVICES=4

train_acordar1() {
    python run_monobert_acordar.py \
        --seed 42 --epoch_num $6 \
        --model_name /path/to/model \
        --train_dir_path ./data/acordar1/data/ \
        --test_dir_path ./data/acordar1/data/ \
        --lr $4 \
        --train_batch_size $5 \
        --gradient_accumulation_steps $5 \
        --task cds_$2_$3 \
        --fold $1
}



for diameter in 4
do
    for edge in 5
    do
        for bs in 16
        do
            for lr in 1e-6 1e-5 3e-5 5e-5
            do
                for epoch in 3 5 10
                do
                    for fold in {0..4}
                    do
                        train_acordar1 $fold $diameter $edge $lr $bs $epoch
                    done
                done
            done
        done
    done
done