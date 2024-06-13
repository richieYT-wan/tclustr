#! /usr/bin/bash

# args : 1 = mainfolder ; 2 = outfolder ; 3 = n_epochs ; 4 = grep

mainfolder=/Users/riwa/Documents/code/tclustr/output/240428_FixTriplet/
outdir=/Users/riwa/Documents/code/tclustr/output/240428_FixTriplet/TCRBASE_ValidSwap/
grepst=$1
grepv=$2
subdir=$(ls -dr ${mainfolder}/*/ | grep -v tcrbase | grep ${grepst} | grep -v ${grepv})
mkdir -p ${outdir}
cd /Users/riwa/Documents/code/tclustr/pyscripts/


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
    # Use grep to find the part of the filename containing "KFold"
    kfolding=$(echo "${name_description}" | grep -o 'KFold_[0-9]*')

    # Use awk to extract the number right after "KFold_"
    kfold_number=$(echo "${kfolding}" | awk -F '_' '{print $2}')

    echo "KFold number: $kfold_number"
    echo "python3 ./do_tcrbase_swaps.py -model_folder "${fullpath}/" -rid ${random_id} -od ${outdir} -o "tcrbase_${name_description}" -db /Users/riwa/Documents/code/tclustr/data/filtered/240418_nettcr_expanded_20binders_17pep_withswaps.csv -kf ${kfold_number}"

    python3 ./do_tcrbase_swaps.py -model_folder "${fullpath}/" -rid ${random_id} -od ${outdir} -o "tcrbase_${name_description}" -db /Users/riwa/Documents/code/tclustr/data/filtered/240418_nettcr_expanded_20binders_17pep_withswaps.csv -kf ${kfold_number}
    echo "####################"
done


