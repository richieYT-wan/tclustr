#!/bin/bash

mainfolder="../output/triplet_tuning/"
subdirs=$(find "${mainfolder}" -maxdepth 1 -mindepth 1 -type d)

for subdir in ${subdirs}; do
    num_directories=$(find "${subdir}" -maxdepth 1 -type d | wc -l)
    
    if [ ${num_directories} -gt 5 ]; then
        echo "Subfolder '${subdir}' has more than 5 directories."
        # If you want to list the directories, uncomment the line below:
        # find "${subdir}" -maxdepth 1 -type d
    fi
done
