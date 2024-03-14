#! /usr/bin/bash


source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate pynn

cd ${CDH}tclustr/pyscripts/


pwd

mainfolders=(231108_TripletCosine_A3B3_margin_01_25k_epochs_larger_model_7VzZ5 231108_TripletCosine_A3B3_margin_Auto_25k_epochs_Ds1PC 231108_TripletL2_A3B3_margin_15_25k_epochs_larger_model_rwUlQ 231129_RedoNoTriplet_CUDA_autoMargin_BeIgZ CDR3_AB_nh_128_wd_1e-4_LOW_KLD_p6da2 CDR3_AB_nh_128_wd_1e-4_weights_3to1_LlWGn)
ids=(7VzZ5 Ds1PC rwUlQ BeIgZ p6da2 LlWGn)
outnames=(Cos_25k_01Large Cos_25k_AutoSmall L2_25k_15Large NoTrp_10k_Small Old_25k_LowKLD_Large Old_25k_HiKLD_Large)


for i in "${!mainfolders[@]}"; do
  mainfolder=${mainfolders[i]}
  id=${ids[i]}
  outname=${outnames[i]}

  for kf in $(seq 0 4); do
    f="../output/VAE_For_CLF/${mainfolder}/*_${kf}_*"
    for folder in $f; do
      if [ -d "$folder" ]; then
        python3 ./train_classifier_frozen_vae.py -cuda True -f ../data/filtered/231205_nettcr_old_26pep_with_swaps.csv -o "CLF_1layer64_025_BN_withSwaps_FLIP_BLOSUM_${outname}" -nh 64 -do 0.25 -bn True -n_layers 1 -lr 1e-4 -wd 1e-4 -bs 1024 -ne 1000 -pepenc BL50LO -kf ${kf} -rid "${id}" -seed ${kf} -model_folder "${folder}/" &
      fi
    done
  done
  wait  # Wait for all background processes to finish before moving to the next mainfolder
done

mkdir -p '../output/FlipBLOSUM/'
mv '../output/*CLF_1layer64_025_BN_withSwaps_FLIP_BLOSUM_*' '../output/FlipBLOSUM/'