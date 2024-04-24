export  CUDA_VISIBLE_DEVICES=4

minehn_ntcir_data() {
python -m FlagEmbedding.baai_general_embedding.finetune.hn_mine \
    --model_name_or_path "/path/to/model" \
    --input_file "./data/ntcir$1/data/cds_$2_$3_split_train.jsonl" \
    --output_file "./data/ntcir$1/data/cds_$2_$3_split_train_minedHN.jsonl" \
    --range_for_sampling "2-200" 
}

for diameter in 3 4
do
    for edge in 5
    do
        for version in 15
        do
            minehn_ntcir_data $version $diameter $edge
        done
    done
done