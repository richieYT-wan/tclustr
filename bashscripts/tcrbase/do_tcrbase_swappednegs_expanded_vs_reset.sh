#! /usr/bin/bash

# args : 1 = mainfolder ; 2 = outfolder ; 3 = n_epochs ; 4 = grep

mainfolder=$1
outdir=$2
grepstatement=$3
grepv=$4
subdir=$(ls -dr ${mainfolder}/*/ | grep "${grepstatement}" | grep -v "${grepv}")
mkdir -p ${outdir}
cd ../../pyscripts/
echo "wtf"
echo "$1 $2"
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

    json_file="${fullpath}$(ls ${fullpath} | grep "checkpoint" | grep "json")"
    best_checkpoint="${fullpath}$(ls ${fullpath} | grep "checkpoint_best" | grep -v "last" | grep ".pt")"
    last_checkpoint="${fullpath}$(ls ${fullpath} | grep "last_epoch" | grep ".pt")"

    echo "KFold number: $kfold_number"
    # Best
    python3 ./do_tcrbase_swaps.py -json_file ${json_file} -pt_file ${best_checkpoint} -rid ${random_id} -od ${2} -o "BEST_tcrbase_${name_description}" -db ../data/multimodal/240326_nettcr_paired_withswaps.csv -kf ${kfold_number}
    # Reset
    python3 ./do_tcrbase_swaps.py -json_file ${json_file} -pt_file ${best_checkpoint} -rid ${random_id} -od ${2} -o "RESET_tcrbase_${name_description}" -db ../data/multimodal/240326_nettcr_paired_withswaps.csv -kf ${kfold_number} -reset True
    # Last
    python3 ./do_tcrbase_swaps.py -json_file ${json_file} -pt_file ${last_checkpoint} -rid ${random_id} -od ${2} -o "LAST_tcrbase_${name_description}" -db ../data/multimodal/240326_nettcr_paired_withswaps.csv -kf ${kfold_number}


    echo "####################"
done


