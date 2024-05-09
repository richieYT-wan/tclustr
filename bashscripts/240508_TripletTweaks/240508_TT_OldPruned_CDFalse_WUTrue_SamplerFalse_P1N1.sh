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
python3 ./231102_fulltcr_tripletloss.py -f /home/projects/vaccine/people/yatwan/tclustr/data/filtered/240507_nettcr_old_pruned_noswap_20peps.csv -pad -20 -enc BL50LO -ne 25000 -cuda True -lwseq 1 -lwkld 1e-2 -lwtrp 3 -dist_type cosine -margin 0.2 -mla1 7 -mla2 8 -mla3 22 -mlb1 6 -mlb2 7 -mlb3 23 -mlpep 0 -nl 100 -nh 256 -bs 1024 -lr 5e-4 -wd 1e-5 -wu 150 -fp 50 -kld_dec 1e-2 -kldts 0.075 -o "${script_name}" -kf 0 -seed 0 -addpe True -bn True -ale True -ald True -ob False -pepweight False -posweight True -dwpos 1 -dwneg 1 -minority_count 50 -wutrp 5000 > "${HOMEDIR}logs/${script_name}.log" 2>&1
