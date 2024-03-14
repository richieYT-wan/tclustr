#! /usr/bin/bash

# args : 1 = mainfolder ; 2 = outfolder ; 3 = n_epochs

mainfolder=$1
outdir=$2
mkdir -p $outdir
subdir=$(ls -dr ${mainfolder}/*/ | grep -vi "clf" | grep -v "PERFx" | grep -v "3omTn\|ws7Ir\|JXuSd\|JXuSd\|IHc02")
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
    if [[ $(ls ../output/${outdir} | grep ${random_id} | wc -l) -ne 1 ]]; then
      for kf in $(seq 0 4);do
        kfolder="${fullpath}*_KFold_${kf}_*/"
        for folder in ${kfolder};do
          if [ -d ${kfolder} ]; then
            python3 ./train_classifier_frozen_vae.py -model_folder "${folder}/" -rid ${random_id} -od ${2}/${folder_name}/ -o "pepweighted_CLF_${name_description}" -cuda True -f ../data/filtered/231205_nettcr_old_26pep_with_swaps.csv -nh 64 -do 0.25 -bn True -n_layers 1 -lr 1e-4 -wd 1e-3 -bs 1024 -ne ${3} -pepenc BL50LO -kf ${kf} -seed ${kf} -pepweight True
          fi
        done
      done
    else
      if [[ $(ls "../output/${outdir}/$(ls ../output/${outdir} | grep ${random_id})" | wc -l) -ne 5 ]];then
        for kf in $(seq 0 4);do
        kfolder="${fullpath}*_KFold_${kf}_*/"
        for folder in ${kfolder};do
          if [ -d ${kfolder} ]; then
            python3 ./train_classifier_frozen_vae.py -model_folder "${folder}/" -rid ${random_id} -od ${2}/${folder_name}/ -o "pepweighted_CLF_${name_description}" -cuda True -f ../data/filtered/231205_nettcr_old_26pep_with_swaps.csv -nh 64 -do 0.25 -bn True -n_layers 1 -lr 1e-4 -wd 1e-3 -bs 1024 -ne ${3} -pepenc BL50LO -kf ${kf} -seed ${kf} -pepweight True
          fi
        done
      done
      fi
    fi
    echo "####################"
done


