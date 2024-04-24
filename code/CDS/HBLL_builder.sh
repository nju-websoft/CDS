base_dir="/path/to/graph"

for folder in "$base_dir"/*; do
    if [[ -d "$folder" ]]; then
        folder_name=$(basename "$folder")
        cd "$folder"
        if ! java -cp code/CBA/CBA/src method.Run ./CBA HBLL UW 2>error_log.txt; then
            echo "Error in folder: $folder_name" | tee -a "$base_dir/error_log.txt"
            cat error_log.txt >> "$base_dir/error_log.txt"
        fi
        echo "$folder_name" >> "$base_dir/HBLL.txt"
        cd "$base_dir"
    fi
done