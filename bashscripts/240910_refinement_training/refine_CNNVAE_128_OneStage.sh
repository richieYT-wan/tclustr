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
outname=RefineTest_ONESTAGE_DELETE

# Run VAE
python3 ./240910_refine_CNNVAE.py -od ${outdir} -o ${outname} -f ${HOMEDIR}/data/OTS/concat_francis_garner_random_5fold.csv -model_pt ${HOMEDIR}/output/240618_NestedKCV_CNNVAE/Nested_TwoStageCNNVAE_NOTRIPLET_ld128_kld_1e-2_ExpData_KFold_0_240730_1232_ph8wm/epoch_4500_interval_checkpoint__kcv_fold_00_Nested_TwoStageCNNVAE_NOTRIPLET_ld128_kld_1e-2_ExpData_KFold_0_240730_1232_ph8wm.pt -model_json ${HOMEDIR}/output/240618_NestedKCV_CNNVAE/Nested_CNNVAE_latent_128_kld_1e-2_ExpData_KFold_0_240618_1607_ER8wJ/checkpoint_best_fold00_kcv_240618_nettcr_exp_nested_posonly_train_p0234_f00_Nested_CNNVAE_latent_128_kld_1e-2_ExpData_KFold_0_240618_1607_ER8wJ_JSON_kwargs.json -model_folder ${HOMEDIR}/output/240618_NestedKCV_CNNVAE/Nested_CNNVAE_latent_128_kld_1e-2_ExpData_KFold_0_240618_1607_ER8wJ/ -device cuda -ne 3000 -kf 0 -debug True -tf ${HOMEDIR}/data/OTS/subsampled_1percent_concat_francis_garner_random_5fold.csv
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
