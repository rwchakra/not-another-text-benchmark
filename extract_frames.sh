#!/bin/bash

# File containing the list of directories
SAMPLED_FILE="${1:-sampled_dirs.txt}"

# Check if the file exists
if [[ ! -f "$SAMPLED_FILE" ]]; then
    echo "Error: $SAMPLED_FILE not found"
    exit 1
fi

# Process each directory
while IFS= read -r dir; do
    # Skip empty lines
    [[ -z "$dir" ]] && continue
    
    # Remove trailing whitespace/commas if any
    dir=$(echo "$dir" | sed 's/,*$//' | xargs)
    
    if [[ ! -d "$dir" ]]; then
        echo "Warning: Directory not found: $dir"
        continue
    fi
    
    echo "Processing: $dir"
    
    # Create input and label directories
    mkdir -p "$dir/input"
    mkdir -p "$dir/label"
    
    # Copy first 8 frames (00-07) to input
    for i in $(seq -w 0 7); do
        # Handle both single digit (00-09) and ensure proper formatting
        frame=$(printf "%02d" $((10#$i)))
        src="$dir/rgb-cam_1-${frame}.jpg"
        if [[ -f "$src" ]]; then
            cp "$src" "$dir/input/"
        else
            echo "  Warning: Missing input frame: $src"
        fi
    done
    
    # Copy next 8 frames (08-15) to label
    for i in $(seq 8 15); do
        frame=$(printf "%02d" $i)
        src="$dir/rgb-cam_1-${frame}.jpg"
        if [[ -f "$src" ]]; then
            cp "$src" "$dir/label/"
        else
            echo "  Warning: Missing label frame: $src"
        fi
    done
    
    echo "  Created input/ with frames 00-07"
    echo "  Created label/ with frames 08-15"
    
done < "$SAMPLED_FILE"

echo "Done!"
