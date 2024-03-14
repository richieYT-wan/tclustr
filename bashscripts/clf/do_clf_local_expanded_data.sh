#! /usr/bin/bash

# The expanded data comes from mathias' new dataset /home/projects/vaccine/people/matjen/expanded_data_project/data/train/nettcr_train_alpha_beta_paired_chain_expanded.csv
mainfolders=(231205_TripletCosine_A3B3_expanded_data_REDO_COSA3B3)
ids=(COSA3B3)
outnames=(Cos_25k_more_data)

cd ../pyscripts/
for i in "${!mainfolders[@]}"; do
  mainfolder=${mainfolders[i]}
  id=${ids[i]}
  outname=${outnames[i]}
  i=0;
  # shellcheck disable=SC2045
  for f in $(ls "../output/TripletTest/${mainfolder}"); do

    python3 ./train_classifier_frozen_vae.py -cuda True -f ../data/filtered/231205_nettcr_old_26pep_with_swaps.csv -o "CLF_Nh50_Do025_True_nl1_ne1000_withSwaps_${outname}" -nh 50 -do 0.25 -bn True -n_layers 1 -lr 1e-4 -wd 1e-3 -bs 1024 -ne 1000 -kf ${i} -rid "${id}" -seed ${i} -model_folder "../output/TripletTest/${mainfolder}/${f}/"; ((i += 1));
  done
done

for i in "${!mainfolders[@]}"; do
  mainfolder=${mainfolders[i]}
  id=${ids[i]}
  outname=${outnames[i]}
  i=0;
  # shellcheck disable=SC2045
  for f in $(ls "../output/TripletTest/${mainfolder}"); do

    python3 ./train_classifier_frozen_vae.py -cuda True -f ../data/filtered/231212_nettcr_expanded_pairedAB_412peps_with_swaps.csv -o "CLF_Nh50_Do025_True_nl1_ne1000_EXPANDED_DATA_withSwaps_${outname}" -nh 50 -do 0.25 -bn True -n_layers 1 -lr 1e-4 -wd 1e-3 -bs 1024 -ne 1000 -kf ${i} -rid "${id}" -seed ${i} -model_folder "../output/TripletTest/${mainfolder}/${f}/"; ((i += 1));
  done
done