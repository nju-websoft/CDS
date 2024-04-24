#! /bin/bash

export  CUDA_VISIBLE_DEVICES=3


test_acordar1_data() {
    python run_monobert_acordar.py \
        --seed 42 --epoch_num $6 \
        --model_name /path/to/model \
        --train_dir_path ./data/acordar1/data/ \
        --test_dir_path ./data/acordar1/data/ \
        --init_checkpoint True \
        --checkpoint_dir ./outputs/acordar/monoBERT_cds_$2_$3_monobert/fold_$1_lr_$4_bs_$5_epoch_$6 \
        --only_eval True \
        --lr $4 \
        --train_batch_size $5 \
        --task cds_$2_$3 \
        --fold $1 \
        --topk $7
}


for diameter in 4
    do
        for edge in 5
        do
            for bs in 16
            do
                for lr in 1e-06 1e-05 3e-05
                do
                    for epoch in 3 5 10
                    do
                        for fold in {0..4}
                        do
                            for topk in 10
                            do
                                test_acordar1_data $fold $diameter $edge $lr $bs $epoch $topk
                            done
                        done
                    done
                done
            done
        done
    done
