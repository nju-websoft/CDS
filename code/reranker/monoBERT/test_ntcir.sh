#! /bin/bash

export  CUDA_VISIBLE_DEVICES=4

test_ntcir_data() {
    python run_monobert_nticr_e.py \
        --seed 42 --epoch_num $6 \
        --model_name /path/to/model \
        --test_path ./data/ntcir$4/data/BM25_top10_cds_$1_$5_split_test.json \
        --dev_path ./data/ntcir$4/data/BM25_top10_cds_$1_$5_split_dev.json \
        --init_checkpoint True \
        --init_checkpoint_path ./outputs/ntcir$4/monoBERT_cds_$1_$5_monobert/lr_$2_bs_$3_epoch_$6/pytorch_model.bin \
        --results_save_path ./results/ntcir$4 \
        --only_eval True \
        --task cds_$1_$5 \
        --lr $2 \
        --train_batch_size $3 \
        --topk $7
}

# for topk in 10 100
# do
#     test_ntcir_metadata $topk
# done

for version in 15
do
    for diameter in 4
    do
        for edge in 5
        do
            for bs in 16
            do
                for lr in 1e-06 1e-05 3e-05 5e-05
                do
                    for epoch in 5 10
                    do
                        for topk in 10
                        do
                            test_ntcir_data $diameter $lr $bs $fold $edge $epoch $topk
                        done
                    done
                done
            done
        done
    done
done