base_dir="/path/to/graph"
for folder in "$base_dir"/*; do
    if [[ -d "$folder" ]]; then
        folder_name=$(basename "$folder")
        if [[ -f "$folder/query.txt" ]]; then
            while IFS= read -r line; do
                x=$(echo "$line" | awk '{print $1}')
                target_folder=$(find "$folder" -type d -name "${x}_0.55")
                if [[ -d "$target_folder" ]]; then
                    cd "$target_folder"
                    java -cp /code/CBA/CBA/src method.Run ./ CBA+ UW 2> error.log
                    if [[ $? -ne 0 ]]; then
                        echo "$folder_name" >> ../CBA.txt
                        echo "Java command failed in $target_folder" >> ../error.log
                    fi
                    cd "$base_dir"
                fi
            done < "$folder/query.txt"
            echo "$folder_name" >> CBA.txt
        fi
    fi
done