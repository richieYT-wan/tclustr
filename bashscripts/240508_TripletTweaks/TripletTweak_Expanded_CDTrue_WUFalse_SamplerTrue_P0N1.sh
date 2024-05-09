#! /usr/bin/bash
HOMEDIR=/home/projects/vaccine/people/yatwan/tclustr/
PYDIR=${HOMEDIR}pyscripts/

source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate cuda

# Get the full path of the script
script_path="$(readlink -f "$0")"

# Extract the basename of the script
script_name="$(basename "$script_path" .sh)"

echo "The basename of the script is: $script_name"



cd ${PYDIR}
python3 ./231102_fulltcr_tripletloss.py -f /home/projects/vaccine/people/yatwan/tclustr/data/multimodal/240326_nettcr_paired_NOswaps.csv -pad -20 -enc BL50LO -ne 25000 -cuda True -lwseq 1 -lwkld 1e-2 -lwtrp 3 -dist_type cosine -margin 0.2 -mla1 7 -mla2 8 -mla3 22 -mlb1 6 -mlb2 7 -mlb3 23 -mlpep 0 -nl 100 -nh 256 -bs 1024 -lr 5e-4 -wd 1e-5 -wu 150 -fp 50 -kld_dec 1e-2 -kldts 0.075 -o "TripletTweak_Expanded_CDTrue_WUFalse_SamplerTrue_P0N1" -kf 0 -seed 0 -addpe True -bn True -ale True -ald True -ob False -pepweight False -posweight True -dwpos 0 -dwneg 1 -minority_count 50 -minority_sampler True -cdtrp 15000 > "${HOMEDIR}logs/TripletTweak_Expanded_CDTrue_WUFalse_SamplerTrue_P0N1.log" 2>&1