#! /usr/bin/bash


mainfolder=$1
subdir=$(ls -d ${mainfolder}/*/ | grep -vi "clf")
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

    for kf in $(seq 0 4);do
      kfolder="${fullpath}*_KFold_${kf}_*/"
      for folder in ${kfolder};do
        if [ -d ${kfolder} ]; then
          echo "kfolder, kf"
          echo "${kf} ${folder}"
          python3 ./train_classifier_frozen_vae.py -model_folder "${folder}/" -rid ${random_id} -od classifier_flipped_blosum/${folder_name}/ -o "CLF_${name_description}" -cuda True -f ../data/filtered/231205_nettcr_old_26pep_with_swaps.csv -nh 64 -do 0.25 -bn True -n_layers 1 -lr 1e-4 -wd 1e-3 -bs 1024 -ne 750 -pepenc BL50LO -kf ${kf} -seed ${kf} 
        fi
      done
    done
    echo "####################"
done


