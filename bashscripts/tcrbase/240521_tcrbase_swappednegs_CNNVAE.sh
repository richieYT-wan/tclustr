#!/bin/bash

# args : 1 = mainfolder ; 2 = outfolder ; 3 = n_epochs ; 4 = grep

mainfolder=/Users/riwa/Documents/code/tclustr/output/240516_CNNVAE_DiffDatasets/
outdir=/Users/riwa/Documents/code/tclustr/output/240516_CNNVAE_DiffDatasets/TCRBASE_ValidSwap/
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
    # Condition if the name matches old
    if [[ "$name_description" =~ OldAll|Old20peps ]]; then
      python3 ./do_tcrbase_swaps.py -model_folder "${fullpath}/" -rid ${random_id} -od ${outdir} -o "tcrbase_${name_description}_old_pruned_20_swaps" -conv True -db /Users/riwa/Documents/code/tclustr/data/filtered/240507_nettcr_old_pruned_wswap_20peps.csv -kf ${kfold_number}

      python3 ./do_tcrbase_swaps.py -model_folder "${fullpath}/" -rid ${random_id} -od ${outdir} -o "tcrbase_${name_description}_old_full" -conv True -db /Users/riwa/Documents/code/tclustr/data/filtered/231205_nettcr_old_26pep_with_swaps.csv -kf ${kfold_number}

    elif [[ "$name_description" =~ ExpAll|78peps ]]; then
      python3 ./do_tcrbase_swaps.py -model_folder "${fullpath}/" -rid ${random_id} -od ${outdir} -o "tcrbase_${name_description}_exp_full" -conv True -db /Users/riwa/Documents/code/tclustr/data/filtered/240326_nettcr_exp_paired_withswaps.csv -kf ${kfold_number}

      python3 ./do_tcrbase_swaps.py -model_folder "${fullpath}/" -rid ${random_id} -od ${outdir} -o "tcrbase_${name_description}_exp_pruned_78peps" -conv True -db /Users/riwa/Documents/code/tclustr/data/filtered/240507_nettcr_exp_pruned_wswap_78peps.csv -kf ${kfold_number}

      python3 ./do_tcrbase_swaps.py -model_folder "${fullpath}/" -rid ${random_id} -od ${outdir} -o "tcrbase_${name_description}_exp_17pep_20binders" -conv True -db /Users/riwa/Documents/code/tclustr/data/filtered/240418_nettcr_expanded_20binders_17pep_withswaps.csv -kf ${kfold_number}

    else
      echo "no match found"
    fi

    echo "####################"
done


