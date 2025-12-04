#!/bin/zsh

# ==============================================================================
# Script Name: convert_levels.sh
# Description: Sets standard_name attribute and interpolates ICON model levels 
#              to pressure levels for all NetCDF files in a directory.
# Usage:       ./convert_levels.sh [path_to_directory]
# ==============================================================================

# Exit immediately if a command exits with a non-zero status
set -e

# Check if CDO is installed
if ! command -v cdo &> /dev/null; then
    echo "Error: 'cdo' is not installed or not in your PATH."
    echo "You can install it via Homebrew: brew install cdo"
    exit 1
fi

# 1. Define Directory
# Use the first argument as the target directory, or default to current directory
INPUT_DIR="${1:-.}"
OUTPUT_DIR="${INPUT_DIR}/pressure_levels_output"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "Processing files in: $INPUT_DIR"
echo "Saving output to:  $OUTPUT_DIR"
echo "-----------------------------------------------------"

# 2. Define Pressure Levels
# CDO ap2pl expects Pascals (Pa). 
# Your list is in hPa, so we multiply by 100.
# List: 1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000
LEVELS_PA="100,200,300,500,700,1000,2000,3000,5000,7000,10000,12500,15000,17500,20000,22500,25000,30000,35000,40000,45000,50000,55000,60000,65000,70000,75000,77500,80000,82500,85000,87500,90000,92500,95000,97500,100000"

# 3. Iterate through files
# We use find to handle filenames with spaces correctly
find "$INPUT_DIR" -maxdepth 1 -name "*.nc" -print0 | while IFS= read -r -d '' FILE; do
    
    BASENAME=$(basename "$FILE")
    OUTPUT_FILE="$OUTPUT_DIR/${BASENAME%.nc}_pl.nc"
    TMP_FILE="$OUTPUT_DIR/${BASENAME%.nc}_tmp.nc"

    echo "Processing: $BASENAME"

    # 4. Execute CDO Command
    # We chain the operators to avoid creating temporary files.
    # Logic: 
    #   1. Input file -> 
    #   2. setattribute (fix pres name) -> 
    #   3. ap2pl (interpolate using the fixed pres variable) -> 
    #   4. Output file
    
    # Note: -P 4 tells CDO to use 4 CPU threads (speeds up interpolation)
    cdo -setattribute,pres@standard_name='air_pressure' "$FILE" "$TMP_FILE"
    cdo ap2pl,"$LEVELS_PA" "$TMP_FILE" "$OUTPUT_FILE"

    if [ $? -eq 0 ]; then
        echo "  -> Success: Created ${BASENAME%.nc}_pl.nc"
    else
        echo "  -> Failed"
    fi

done

echo "-----------------------------------------------------"
echo "All done. Files available in $OUTPUT_DIR"