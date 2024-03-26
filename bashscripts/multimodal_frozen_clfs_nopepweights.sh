#! /bin/bash

# args : 1 = mainfolder ; 2 = outfolder ; 3 = n_epochs ; 4 = grep

mainfolder='../output/mmvae_models'
outdir='../output/mmvae_clfs_output/'
mkdir -p ${outdir}
n_epochs=$3
grepstatement=$4
#subdir=("${mainfolder}"/*/)
subdir=$(ls -dr ${mainfolder}/*/)
# Use an array to capture the output
cd ../pyscripts/

# Use quoted expansion with @ to iterate over the array properly
for f in ${subdir}; do
#    echo "xx${f}xx"
    folder_name=$(basename "${f}")
#    echo "#######################################"
#    echo "HERE"
#    echo ${folder_name}
#    echo ${f}
#    echo "#######################################"
    # Split folder_name at the last underscore
    name_description=$(echo "${folder_name%_*}")
    random_id=$(echo "${folder_name##*_}")
    # Print or use the extracted values as needed
        # Use grep to find the part of the filename containing "KFold"
    kfolding=$(echo "${name_description}" | grep -o 'KFold_[0-9]*')

    # Use awk to extract the number right after "KFold_"
    kfold_number=$(echo "${kfolding}" | awk -F '_' '{print $2}')
    echo "------------------------------------------------------------------------"
    echo "************************"
    echo "Name: ${name_description}"
    echo "Random ID: ${random_id}"
    echo "filepath: ${f}"
    echo "KFold number: $kfold_number"
    echo "************************"
    json_file="${f}$(ls ${f} | grep "checkpoint" | grep "json")"
    best_checkpoint="${f}$(ls ${f} | grep "checkpoint_best" | grep -v "last" | grep ".pt")"
    last_checkpoint="${f}$(ls ${f} | grep "last_epoch" | grep ".pt")"
    echo ${json_file}
    echo ${best_checkpoint}
    echo ${last_checkpoint}
    # Doing with "best" checkpoint
    python3 ./train_classifier_frozen_mmvae.py -json_file ${json_file} -pt_file ${best_checkpoint} -rid ${random_id} -od "${2}/BEST_${folder_name}/" -o "BEST_${name_description}" -cuda True -f ../data/multimodal/240326_nettcr_paired_withswaps.csv -nh 64 -do 0.25 -bn True -n_layers 1 -lr 1e-4 -wd 1e-5 -bs 1024 -ne ${3} -pepenc 'none' -pepweight False -kf ${kfold_number} -seed ${kfold_number}
    # Doing with "last" checkpoint
    python3 ./train_classifier_frozen_mmvae.py -json_file ${json_file} -pt_file ${last_checkpoint} -rid ${random_id} -od "${2}/LAST_${folder_name}/" -o "LAST_${name_description}" -cuda True -f ../data/multimodal/240326_nettcr_paired_withswaps.csv -nh 64 -do 0.25 -bn True -n_layers 1 -lr 1e-4 -wd 1e-5 -bs 1024 -ne ${3} -pepenc 'none' -pepweight False -kf ${kfold_number} -seed ${kfold_number}
    echo "------------------------------------------------------------------------"
    echo ""
done


