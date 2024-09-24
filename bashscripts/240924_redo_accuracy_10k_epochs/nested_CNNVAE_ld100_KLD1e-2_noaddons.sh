source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate cuda

HOMEDIR=/home/projects/vaccine/people/yatwan/tclustr/
outdir="240924_redoAccuracies"
RESDIR="${HOMEDIR}output/${outdir}"
PYDIR=${HOMEDIR}pyscripts/
cd ${PYDIR}

# Define the characters that can be used
characters="abcdefghijkmnopqrstuvwxyzABCDEFGHJKLMNOPQRSTUVWXYZ0123456789"
# Generate a random index between 0 and 61 (total number of characters)
index=$((RANDOM % 60))
# Get the character at the generated index
first_char="${characters:index:1}"
# Generate the remaining 4 characters as a combination of the defined characters
rest_chars=$(head /dev/urandom | tr -dc "$characters" | head -c 4)
# Combine the first and remaining characters
random_id="${first_char}${rest_chars}"
outname=ld100_KLD1e-2_no_addons

# Set the flat phase to 8k epochs and warm-up to 0 and decrease to 0
python3 ./240515_cnnvae_tripletloss.py -f ${HOMEDIR}data/filtered/240618_nettcr_exp_nested_posonly_train_p0234.csv -od ${outdir} -pad -20 -enc BL50LO -ne 10000 -cuda True -lwseq 1 -lwkld 1d-2 -lwtrp 0 -mla1 7 -mla2 8 -mla3 22 -mlb1 6 -mlb2 7 -mlb3 23 -mlpep 0 -nl 100 -nh 100 -bs 512 -lr 1e-4 -wd 1e-4 -wu 0 -fp 10000 -kld_dec 0 -kldts 0.075 -o ${outname} -kf 0 -seed 0 -addpe False -posweight False -bn True -pepweight False -rid ${random_id} -tf ${HOMEDIR}data/filtered/240618_nettcr_exp_nested_posonly_test_p1.csv
