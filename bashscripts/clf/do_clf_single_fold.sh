#! /usr/bin/bash

# args : 1 = mainfolder ; 2 = outfolder ; 3 = n_epochs ; 4 = grep

mainfolder=$1
outdir=$2
grepstatement=$4
subdir=$(ls -dr ${mainfolder}/*/ | grep -vi "clf" | grep -v "PERFx" | grep -v "3omTn\|ws7Ir\|JXuSd\|JXuSd\|IHc02" | grep "${grepstatement}")
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
    python3 ./train_classifier_frozen_vae.py -model_folder "${fullpath}/" -rid ${random_id} -od ${2}/${folder_name}/ -o "CLF_${name_description}" -cuda True -f ../data/filtered/231205_nettcr_old_26pep_with_swaps.csv -nh 64 -do 0.25 -bn True -n_layers 1 -lr 1e-4 -wd 1e-3 -bs 1024 -ne ${3} -pepenc BL50LO -kf ${kfold_number} -seed ${kfold_number}
    echo "####################"

    python3 ./train_classifier_frozen_mmvae.py -json_file ${json_file} -pt_file ${best_checkpoint} -rid ${random_id} -od "${outdir}/BEST_${folder_name}/" -o "BEST_${name_description}" -cuda True -f ../data/multimodal/240326_nettcr_paired_withswaps.csv -nh 64 -do 0.25 -bn True -n_layers 1 -lr 1e-4 -wd 5e-6 -bs 2048 -ne ${n_epochs} -pepenc 'none' -pepweight False -kf ${kfold_number} -seed ${kfold_number}
done


