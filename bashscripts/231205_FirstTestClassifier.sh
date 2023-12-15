#! /usr/bin/bash

#!/bin/bash

# TODO: ARGS
# TODO: 1 : walltime


# Define the characters that can be used
characters="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
# Generate a random index between 0 and 61 (total number of characters)
index=$((RANDOM % 62))
# Get the character at the generated index
first_char="${characters:index:1}"
# Generate the remaining 4 characters as a combination of the defined characters
rest_chars=$(head /dev/urandom | tr -dc "$characters" | head -c 4)
# Combine the first and remaining characters
random_string="${first_char}${rest_chars}"
outname="231205_ClassifierFirstTest"



source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate cuda

HOMEDIR=/home/projects/vaccine/people/yatwan/tclustr/
PYDIR=${HOMEDIR}pyscripts/
filename=${filename}
cd ${PYDIR}
python3 ./train_classifier_frozen_vae.py -cuda True -f ../data/filtered/231205_nettcr_old_26pep_with_swaps.csv -o ${outname} -nh 32 -do 0.15 -bn False -n_layers 1 -lr 1e-4 -wd 1e-4 -bs 512 -ne 2500 -kf 0 -rid ${random_string} -seed 0 -model_folder ../output/TripletTest/231108_TripletCosine_A3B3_margin_Auto_25k_epochs_larger_model_7VzZ5/231108_TripletCosine_A3B3_margin01_25k_epochs_larger_model_KFold_0_231130_2321_5PXfp/
