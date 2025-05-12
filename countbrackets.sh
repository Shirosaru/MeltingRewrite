#!/bin/bash

# Ensure a file is provided as an argument
if [ -z "$1" ]; then
    echo "Usage: $0 <file>"
    exit 1
fi

# Count the number of occurrences of '[' and ']' in the file
count_open=$(grep -o "\[" "$1" | wc -l)
count_close=$(grep -o "\]" "$1" | wc -l)

# Total count of square brackets
total_count=$((count_open + count_close))

echo "Number of square brackets: $total_count"
