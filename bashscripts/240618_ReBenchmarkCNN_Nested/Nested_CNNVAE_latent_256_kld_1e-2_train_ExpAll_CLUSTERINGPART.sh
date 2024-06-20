source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate cuda

HOMEDIR=/home/projects/vaccine/people/yatwan/tclustr/
outdir="240618_NestedKCV_CNNVAE"
RESDIR="${HOMEDIR}output/${outdir}"
PYDIR=${HOMEDIR}pyscripts/
cd ${PYDIR}

random_id="vnN02"
outname=Nested_CNNVAE_latent_256_kld_1e-2_ExpData

outmatch=$(ls -t ${RESDIR} | grep ${random_id} | head -n 1)
iid=ExpData17peps
idf=/home/projects/vaccine/people/yatwan/tclustr/data/filtered/240418_nettcr_expanded_20binders_17pep_POSONLY.csv
# Run clustering part
python3 ./240618_nested_VAE_Clustering_intervals.py -rid ${random_id} -np 500 -kf 0 -if 1 -o ${outname}_${iid} -od ../output/${outdir}/clustering/ -tbcralign /home/projects/vaccine/people/yatwan/tclustr/output/240411_ClusteringTests/dist_matrices/2404XX_OUTPUT_tbcralign_distmatrix_140peps_labeled.csv -tcrdist /home/projects/vaccine/people/yatwan/tclustr/output/240411_ClusteringTests/dist_matrices/tcrdist3_distmatrix_140peps_new_labeled.csv -f ${idf} -model_folder "${RESDIR}/${outmatch}" -n_jobs 40 -dn ${iid}
outmatch=$(ls -t ${RESDIR} | grep ${random_id} | head -n 1)
iid=ExpDataTop78
idf=/home/projects/vaccine/people/yatwan/tclustr/data/filtered/240507_nettcr_exp_pruned_noswap_78peps.csv
# Run clustering part
python3 ./240618_nested_VAE_Clustering_intervals.py -rid ${random_id} -np 500 -kf 0 -if 1 -o ${outname}_${iid} -od ../output/${outdir}/clustering/ -tbcralign /home/projects/vaccine/people/yatwan/tclustr/output/240411_ClusteringTests/dist_matrices/2404XX_OUTPUT_tbcralign_distmatrix_140peps_labeled.csv -tcrdist /home/projects/vaccine/people/yatwan/tclustr/output/240411_ClusteringTests/dist_matrices/tcrdist3_distmatrix_140peps_new_labeled.csv -f ${idf} -model_folder "${RESDIR}/${outmatch}" -n_jobs 40 -dn ${iid}
