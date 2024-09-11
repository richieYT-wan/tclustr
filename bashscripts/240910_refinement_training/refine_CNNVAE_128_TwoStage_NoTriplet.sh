source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate cuda

HOMEDIR=/home/projects/vaccine/people/yatwan/tclustr/
outdir="240910_RefineTest_DELETEME"
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
random_id=DELETE
outname=RefineTest_TWOSTAGE_NoTriplet
modelfolder="${HOMEDIR}/output/240618_NestedKCV_CNNVAE/Nested_TwoStageCNNVAE_NOTRIPLET_ld128_kld_1e-2_ExpData_BLOSUM_PEP_KFold_0_240910_1436_N1jMC/"
json=$(ls "${modelfolder}*JSON_kwargs*.json")
pt=$(ls "${modelfolder}*4500*.pt")
# Run VAE
python3 ./240910_refine_CNNVAE.py -od ${outdir} -o ${outname} -f ${HOMEDIR}/data/OTS/concat_francis_garner_random_5fold.csv -model_pt ${pt} -model_json ${json} -model_folder ${modelfolder} -device cuda -ne 2500 -kf 0 -debug False -tf ${HOMEDIR}/data/OTS/subsampled_1percent_concat_francis_garner_random_5fold.csv
#
#outmatch=$(ls -t ${RESDIR} | grep ${random_id} | head -n 1)
#iid=ExpData17peps
#idf=/home/projects/vaccine/people/yatwan/tclustr/data/filtered/240418_nettcr_expanded_20binders_17pep_POSONLY.csv
## Run clustering part
#python3 ./240618_nested_VAE_Clustering_intervals.py -rid ${random_id} -np 500 -kf 0 -if 1 -o ${outname}_${iid} -od ../output/${outdir}/clustering/ -tbcralign /home/projects/vaccine/people/yatwan/tclustr/output/240411_ClusteringTests/dist_matrices/2404XX_OUTPUT_tbcralign_distmatrix_140peps_labeled.csv -tcrdist /home/projects/vaccine/people/yatwan/tclustr/output/240411_ClusteringTests/dist_matrices/tcrdist3_distmatrix_140peps_new_labeled.csv -f ${idf} -model_folder "${RESDIR}/${outmatch}" -rb True -n_jobs 40 -dn ${iid} -bf ../output/240515_IntervalClustering
#outmatch=$(ls -t ${RESDIR} | grep ${random_id} | head -n 1)
#iid=ExpDataTop78
#idf=/home/projects/vaccine/people/yatwan/tclustr/data/filtered/240507_nettcr_exp_pruned_noswap_78peps.csv
## Run clustering part
#python3 ./240618_nested_VAE_Clustering_intervals.py -rid ${random_id} -np 500 -kf 0 -if 1 -o ${outname}_${iid} -od ../output/${outdir}/clustering/ -tbcralign /home/projects/vaccine/people/yatwan/tclustr/output/240411_ClusteringTests/dist_matrices/2404XX_OUTPUT_tbcralign_distmatrix_140peps_labeled.csv -tcrdist /home/projects/vaccine/people/yatwan/tclustr/output/240411_ClusteringTests/dist_matrices/tcrdist3_distmatrix_140peps_new_labeled.csv -f ${idf} -model_folder "${RESDIR}/${outmatch}" -rb True -n_jobs 40 -dn ${iid} -bf ../output/240515_IntervalClustering
