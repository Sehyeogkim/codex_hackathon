#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 OUTPUT_DIRECTORY" >&2
  exit 2
fi

output_directory=$1
mkdir -p "$output_directory"

archive_path="$output_directory/subject-07.tar.gz"
if [[ -f "$archive_path" ]]; then
  echo "DexYCB archive already exists: $archive_path"
  exit 0
fi

# The official Google Drive file regularly reaches its public quota. This
# public Hugging Face mirror serves the full subject-07 archive with the RGB
# streams and metadata needed by this demo. Discovery has verified two
# distinct right-hand mustard-bottle sequences in subject-07.
# huggingface_hub uses the repository's Xet backend and adaptive concurrency.
python -m pip install --quiet --upgrade 'huggingface_hub>=0.32'
source_name='20200928-subject-07.tar.gz'
mem_kib=$(awk '/MemTotal:/ {print $2}' /proc/meminfo 2>/dev/null || echo 0)
if (( mem_kib >= 67108864 )); then
  export HF_XET_HIGH_PERFORMANCE=1
fi
hf download UCBProject/DexYCB "$source_name" \
  --repo-type dataset --local-dir "$output_directory"
mv "$output_directory/$source_name" "$archive_path"

echo "Downloaded DexYCB subject-07 from UCBProject/DexYCB to $archive_path"
