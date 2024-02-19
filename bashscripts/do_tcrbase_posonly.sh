#! /usr/bin/bash

# args : 1 = mainfolder ; 2 = outfolder ; 3 = n_epochs ; 4 = grep

mainfolder=$1
outdir=$2
grepstatement=$3
subdir=$(ls -dr ${mainfolder}/*/ | grep "${grepstatement}")
cd ../pyscripts/
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
    python3 ./do_tcrbase.py -model_folder "${fullpath}/" -rid ${random_id} -od ${2} -o "tcrbase_${name_description}" -db ../data/filtered/230927_nettcr_positives_only.csv -kf ${kfold_number}
    echo "####################"
done


