#!/bin/bash

# Define the paths

host_home_path=$(eval echo "~$USER")
host_lpdata_folder="${HOST_HOME_PATH}/.lpData"
conda_env_dir="/models--yerfor--Real3DPortrait/anaconda3"
model_path_prefix="/models"

# Function to rewrite paths in a single file
rewrite_paths_in_file() {
    local file_path="$1"
    
    sed -i "s|$host_lpdata_path|$model_path_prefix|g" "$file_path"
    echo "Rewritten paths in: $file_path"
}

export -f rewrite_paths_in_file
export host_home_path
export container_path_prefix

# Recursively find and process all files under the conda_env_dir using xargs
# find "$conda_env_dir" -type f finds all files in the directory.
# xargs -n 1 -P "$(nproc)" -I {} bash -c 'rewrite_paths_in_file "$@"' _ {} runs rewrite_paths_in_file in parallel.
# -n 1 tells xargs to use one argument per command invocation.
# -P "$(nproc)" specifies the number of parallel processes to run, using the number of CPUs available on the system ($(nproc)).
find "${host_lpdata_path}${model_path_prefix}${conda_env_dir}" -type f | xargs -n 1 -P "$(nproc)" -I {} bash -c 'rewrite_paths_in_file "$@"' _ {}

echo "Path rewriting completed."
