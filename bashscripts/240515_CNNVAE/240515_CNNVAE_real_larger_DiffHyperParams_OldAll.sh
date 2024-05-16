source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate cuda

HOMEDIR=/home/projects/vaccine/people/yatwan/tclustr/
PYDIR=${HOMEDIR}pyscripts/
# Get the full path of the script
script_path="$(readlink -f "$0")"

# Extract the basename of the script
script_name="$(basename "$script_path" .sh)"

echo "The basename of the script is: $script_name"

cd ${PYDIR}
python3 ./240515_cnnvae_tripletloss.py -f /home/projects/vaccine/people/yatwan/tclustr/data/filtered/240416_nettcr_old_26pep_no_swaps.csv -pad -20 -enc BL50LO -ne 30000 -cuda True -lwseq 1 -lwkld 1e-1 -lwtrp 10 -dist_type cosine -margin 0.2 -mla1 7 -mla2 8 -mla3 22 -mlb1 6 -mlb2 7 -mlb3 23 -mlpep 0 -nl 128 -nh 256 -bs 512 -lr 2.5e-4 -wd 1e-4 -wu 150 -fp 150 -kld_dec 1e-3 -kldts 0.075 -o "CNNVAE_OldAll_KF0_GroupSamplerLARGER_KL1e-1_LWT10_30kepochs" -kf 0 -seed 0 -addpe True -bn True -pepweight False -posweight True -minority_sampler True -minority_count 50 > "${HOMEDIR}logs/240515_CNNVAE_OldAll_Real_Large_KL1e-1_LWT10_30kepochs.log" 2>&1
