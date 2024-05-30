source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate cuda

HOMEDIR=/home/projects/vaccine/people/yatwan/tclustr/
PYDIR=${HOMEDIR}pyscripts/
cd ${PYDIR}
model_folder=/home/projects/vaccine/people/yatwan/tclustr/output/240516_CNNVAE_DiffDatasets/240516_1703_CNNVAE_78peps_KF0_GroupSamplerLARGER_KL1e-1_LWT10_30kepochs_KFold_0_qJO5cD
tbcralign=/home/projects/vaccine/people/yatwan/tclustr/output/240411_ClusteringTests/dist_matrices/2404XX_OUTPUT_tbcralign_distmatrix_140peps_labeled.csv
tcrdist=/home/projects/vaccine/people/yatwan/tclustr/output/240411_ClusteringTests/dist_matrices/tcrdist3_distmatrix_140peps_new_labeled.csv
iid=ExpData17peps
idf=/home/projects/vaccine/people/yatwan/tclustr/data/filtered/240418_nettcr_expanded_20binders_17pep_POSONLY.csv
python3 ./240420_VAE_Clustering_intervals.py -np 500 -kf 0 -o ${iid}_CNNVAE_78peps_LARGER -od ../output/240530_CNNclust_rerun/ -tbcralign ${tbcralign} -tcrdist ${tcrdist} -f ${idf} -model_folder ${model_folder} -rb True -n_jobs 40 -dn ${iid} -bf ../output/240515_IntervalClustering
