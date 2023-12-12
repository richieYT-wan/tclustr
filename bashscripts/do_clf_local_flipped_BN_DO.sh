#! /usr/bin/bash

# This seems to be better!! to do BN first then DO
mainfolders=(231108_TripletCosine_A3B3_margin_01_25k_epochs_larger_model_7VzZ5 231108_TripletCosine_A3B3_margin_Auto_25k_epochs_Ds1PC 231108_TripletL2_A3B3_margin_15_25k_epochs_larger_model_rwUlQ 231129_RedoNoTriplet_CUDA_autoMargin_BeIgZ CDR3_AB_nh_128_wd_1e-4_LOW_KLD_p6da2 CDR3_AB_nh_128_wd_1e-4_weights_3to1_LlWGn)
ids=(7VzZ5 Ds1PC rwUlQ BeIgZ p6da2 LlWGn)
outnames=(Cos_25k_01Large Cos_25k_AutoSmall L2_25k_15Large NoTrp_10k_Small Old_25k_LowKLD_Large Old_25k_HiKLD_Large)

cd ../pyscripts/
for i in "${!mainfolders[@]}"; do
  mainfolder=${mainfolders[i]}
  id=${ids[i]}
  outname=${outnames[i]}
  i=0;
  # shellcheck disable=SC2045
  for f in $(ls "../output/TripletTest/${mainfolder}"); do

    python3 ./train_classifier_frozen_vae.py -cuda True -f ../data/filtered/231205_nettcr_old_26pep_with_swaps.csv -o "CLF_Nh50_Do025_True_nl1_ne1000_withSwaps_FLIPPED_BNDO_${outname}" -nh 50 -do 0.25 -bn True -n_layers 1 -lr 1e-4 -wd 1e-3 -bs 1024 -ne 1000 -kf ${i} -rid "${id}" -seed ${i} -model_folder "../output/TripletTest/${mainfolder}/${f}/"; ((i += 1));
  done
done

# TODO: Do dual modal training

# TODO ask mathias : Nettcr-pan-trained on my data, and get the test predictions for the 5fold kcv to see benchmark