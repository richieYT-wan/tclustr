#! /usr/bin/bash

mainfolder=$1
outdir=$2

# Extract subdirectories from mainfolder
subdir=$(ls -d ${mainfolder}/*/ | grep -vi "clf" | grep -v "PERFx" | grep -v "3omTn\|ws7Ir\|JXuSd\|JXuSd\|IHc02")

# Extract subdirectories from outdir
existing_subdirs=$(ls -d ${outdir}/*/)

# Create temporary files
tempfile1=$(mktemp)
tempfile2=$(mktemp)

# Write subdirectories to temporary files
echo "$subdir" | tr ' ' '\n' | sort > "$tempfile1"
echo "$existing_subdirs" | tr ' ' '\n' | sort > "$tempfile2"

# Remove common subdirectories
subdir=$(comm -23 "$tempfile1" "$tempfile2" | tr '\n' ' ')

# Print the result
echo "$subdir"

# Remove temporary files
rm -f "$tempfile1" "$tempfile2"
