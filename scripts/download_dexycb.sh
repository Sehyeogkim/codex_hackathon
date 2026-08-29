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

python -m pip install --quiet gdown
gdown --fuzzy \
  'https://drive.google.com/file/d/1oWEYD_o3PVh39pLzMlJcArkDtMj4nzI0/edit' \
  --output "$archive_path"

echo "Downloaded DexYCB subject-07 to $archive_path"
