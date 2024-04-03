#! /usr/bin/bash

# args : 1 = mainfolder ; 2 = outfolder ; 3 = n_epochs ; 4 = grep

mainfolder='../output/vaes_models_retrain_old_expdata'
outdir='../output/240328_Retrained_SVAE_CLFs/'
#grepstatement=$4
subdir=$(ls -dr ${mainfolder}/*/)
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
    json_file="${fullpath}$(ls ${fullpath} | grep "checkpoint" | grep "json")"
    best_checkpoint="${fullpath}$(ls ${fullpath} | grep "checkpoint_best" | grep -v "last" | grep ".pt")"
    last_checkpoint="${fullpath}$(ls ${fullpath} | grep "last_epoch" | grep ".pt")"
    echo "KFold number: $kfold_number"
    python3 ./train_classifier_frozen_vae.py -json_file ${json_file} -pt_file ${best_checkpoint} -rid ${random_id} -od "${outdir}" -o "BEST_${name_description}" -cuda False -f ../data/multimodal/240326_nettcr_paired_withswaps.csv -nh 64 -do 0.25 -bn True -n_layers 1 -lr 1e-4 -wd 5e-6 -bs 2048 -ne 2000 -pepenc BL50LO -kf 0 -seed 0 -pepenc 'none' -pepweight False
    python3 ./train_classifier_frozen_vae.py -json_file ${json_file} -pt_file ${last_checkpoint} -rid ${random_id} -od "${outdir}" -o "LAST_${name_description}" -cuda False -f ../data/multimodal/240326_nettcr_paired_withswaps.csv -nh 64 -do 0.25 -bn True -n_layers 1 -lr 1e-4 -wd 5e-6 -bs 2048 -ne 2000 -pepenc BL50LO -kf 0 -seed 0 -pepenc 'none' -pepweight False
    echo "####################"
done


