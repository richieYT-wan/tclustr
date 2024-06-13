#! /usr/bin/bash

# args : 1 = mainfolder ; 2 = outfolder ; 3 = n_epochs ; 4 = grep
## 240408 : Local output paths have been renamed to 240404_BestLastReset_CLF_comparison
mainfolder='/home/projects/vaccine/people/yatwan/tclustr/output/240404_FirstBestLast_comparison/posonly_vaes'
outdir='/home/projects/vaccine/people/yatwan/tclustr/output/240404_FirstBestLast_comparison/clf_outputs/'
n_epochs=2000
subdir=$(ls -dr ${mainfolder}/*/ | grep "24")

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
    script_content=$(cat <<EOF
source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate cuda

HOMEDIR=/home/projects/vaccine/people/yatwan/tclustr/
PYDIR=\${HOMEDIR}pyscripts/
cd \${PYDIR}

python3 ./train_classifier_frozen_vae.py -json_file ${json_file} -pt_file ${best_checkpoint} -rid ${random_id} -od "${outdir}" -o "BEST_${name_description}" -cuda True -f ../data/multimodal/240326_nettcr_paired_withswaps.csv -nh 64 -do 0.25 -bn True -n_layers 1 -lr 1e-4 -wd 5e-6 -bs 2048 -ne ${n_epochs} -pepenc BL50LO -pepweight False -kf 0 -seed 0
python3 ./train_classifier_frozen_vae.py -json_file ${json_file} -pt_file ${last_checkpoint} -rid ${random_id} -od "${outdir}" -o "LAST_${name_description}" -cuda True -f ../data/multimodal/240326_nettcr_paired_withswaps.csv -nh 64 -do 0.25 -bn True -n_layers 1 -lr 1e-4 -wd 5e-6 -bs 2048 -ne ${n_epochs} -pepenc BL50LO -pepweight False -kf 0 -seed 0
python3 ./train_classifier_frozen_vae.py -json_file ${json_file} -pt_file ${last_checkpoint} -rid ${random_id} -od "${outdir}" -o "RESET_${name_description}" -cuda True -f ../data/multimodal/240326_nettcr_paired_withswaps.csv -nh 64 -do 0.25 -bn True -n_layers 1 -lr 1e-4 -wd 5e-6 -bs 2048 -ne ${n_epochs} -pepenc BL50LO -pepweight False -kf 0 -seed 0 -reset True
EOF
)
    echo "$script_content" > "/home/projects/vaccine/people/yatwan/tclustr/bashscripts/240404_CLF_CompareBestLastRandom/${name_description}.sh"
    chmod +x "/home/projects/vaccine/people/yatwan/tclustr/bashscripts/240404_CLF_CompareBestLastRandom/${name_description}.sh"
    echo "------------------------------------------------------------------------"
    echo ""
done
echo "DONE"


