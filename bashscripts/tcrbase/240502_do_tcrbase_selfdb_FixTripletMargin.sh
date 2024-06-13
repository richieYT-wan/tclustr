#! /usr/bin/bash

# args : 1 = mainfolder ; 2 = outfolder ; 3 = n_epochs ; 4 = grep

mainfolder=/Users/riwa/Documents/code/tclustr/output/240428_FixTriplet/
outdir=/Users/riwa/Documents/code/tclustr/output/240428_FixTriplet/TCRBASE_SelfDB/
grepst=$1
grepv=$2
subdir=$(ls -dr ${mainfolder}/*/ | grep -v tcrbase | grep ${grepst} | grep -v ${grepv})
mkdir -p ${outdir}
cd /Users/riwa/Documents/code/tclustr/pyscripts/

# echo $(ls -dr subdir)

for fullpath in ${subdir}; do
    # Extract inner-most folder name without trailing "/"
    folder_name=$(basename "${fullpath}")
    # Split folder_name at the last underscore
    name_description=$(echo "${folder_name%_*}")
    random_id=$(echo "${folder_name##*_}")
    # Print or use the extracted values as needed
    echo "####################"
    echo "Name: ${name_description}"
    echo "Random ID: ${random_id}"
    echo "filepath: ${fullpath}"

    # Use awk to extract the number right after "KFold_"

    python3 ./do_tcrbase_swaps_self_query_db.py -model_folder "${fullpath}/" -rid ${random_id} -od ${outdir} -o "tcrbase_${name_description}" -f /Users/riwa/Documents/code/tclustr/data/filtered/240418_nettcr_expanded_20binders_17pep_POSONLY.csv 
    echo "####################"
done


