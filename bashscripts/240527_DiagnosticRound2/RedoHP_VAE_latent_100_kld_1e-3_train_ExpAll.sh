source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate cuda

HOMEDIR=/home/projects/vaccine/people/yatwan/tclustr/
outdir="240527_DiagnosticRound2"
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
outname=RedoHP_VAE_latent_100_kld_1e-3_train_ExpAll

# Run VAE
python3 ./231102_fulltcr_tripletloss.py -f ${HOMEDIR}data/filtered/240326_nettcr_paired_NOswaps.csv -od ${outdir} -pad -20 -enc BL50LO -ne 20000 -cuda True -lwseq 1 -lwkld 1e-3 -lwtrp 3 -dist_type cosine -margin 0.2 -mla1 7 -mla2 8 -mla3 22 -mlb1 6 -mlb2 7 -mlb3 23 -mlpep 0 -nl 100 -nh 128 -bs 512 -lr 1e-4 -wd 1e-4 -wu 150 -fp 50 -kld_dec 1e-2 -kldts 0.075 -o ${outname} -kf 0 -seed 0 -addpe True -bn True -ale True -ald True -ob False -pepweight False -posweight True -rid ${random_id}
outmatch=$(ls -t ${RESDIR} | grep ${random_id} | head -n 1)
iid=ExpData17peps
idf=/home/projects/vaccine/people/yatwan/tclustr/data/filtered/240418_nettcr_expanded_20binders_17pep_POSONLY.csv
# Run clustering part
python3 ./240420_VAE_Clustering_intervals.py -rid ${random_id} -np 500 -kf 0 -o ${outname}_${iid} -od ../output/${outdir}/clustering/ -tbcralign /home/projects/vaccine/people/yatwan/tclustr/output/240411_ClusteringTests/dist_matrices/2404XX_OUTPUT_tbcralign_distmatrix_140peps_labeled.csv -tcrdist /home/projects/vaccine/people/yatwan/tclustr/output/240411_ClusteringTests/dist_matrices/tcrdist3_distmatrix_140peps_new_labeled.csv -f ${idf} -model_folder "${RESDIR}/${outmatch}" -rb True -n_jobs 40 -dn ${iid} -bf ../output/240515_IntervalClustering
outmatch=$(ls -t ${RESDIR} | grep ${random_id} | head -n 1)
iid=ExpDataTop78
idf=/home/projects/vaccine/people/yatwan/tclustr/data/filtered/240507_nettcr_exp_pruned_noswap_78peps.csv
# Run clustering part
python3 ./240420_VAE_Clustering_intervals.py -rid ${random_id} -np 500 -kf 0 -o ${outname}_${iid} -od ../output/${outdir}/clustering/ -tbcralign /home/projects/vaccine/people/yatwan/tclustr/output/240411_ClusteringTests/dist_matrices/2404XX_OUTPUT_tbcralign_distmatrix_140peps_labeled.csv -tcrdist /home/projects/vaccine/people/yatwan/tclustr/output/240411_ClusteringTests/dist_matrices/tcrdist3_distmatrix_140peps_new_labeled.csv -f ${idf} -model_folder "${RESDIR}/${outmatch}" -rb True -n_jobs 40 -dn ${iid} -bf ../output/240515_IntervalClustering
