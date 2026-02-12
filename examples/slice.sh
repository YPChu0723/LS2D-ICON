#!/bin/zsh

# Define your directories for clarity
IN_DIR="/Users/yunpeichu/CLOUDLAB_MIP/Data/CLOUDLAB_MIP_input/ICON_reglatlon"
OUT_DIR="$IN_DIR/sliced"

# 1. Create the output directory if it doesn't exist
mkdir -p "$OUT_DIR"

for file in "$IN_DIR"/*.nc; do
    # 2. Extract just the filename (e.g., "mydata.nc") from the full path
    filename=$(basename "$file")
    
    echo "Processing $filename..."
    
    ncks -O \
         -d height_3,34,80 \
         -d height_2,34,80 \
         -d height,34,79 \
         "$file" "$OUT_DIR/$filename"
done