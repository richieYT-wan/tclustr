source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate cuda

HOMEDIR=/home/projects/vaccine/people/yatwan/tclustr/
PYDIR=${HOMEDIR}pyscripts/
cd ${PYDIR}
model_folder=/home/projects/vaccine/people/yatwan/tclustr/output/240508_TripletTweaks/240509_1205_TripletTweak_Expanded_CDTrue_WUTrue_SamplerFalse_P1N1_KFold_0_FzWxAi
tbcralign=/home/projects/vaccine/people/yatwan/tclustr/output/240411_ClusteringTests/dist_matrices/2404XX_OUTPUT_tbcralign_distmatrix_140peps_labeled.csv
tcrdist=/home/projects/vaccine/people/yatwan/tclustr/output/240411_ClusteringTests/dist_matrices/tcrdist3_distmatrix_140peps_new_labeled.csv
iid=Exp78Peps
idf=/home/projects/vaccine/people/yatwan/tclustr/data/filtered/240507_nettcr_exp_pruned_noswap_78peps.csv
python3 ./240420_VAE_Clustering_intervals.py -np 500 -kf 0 -o TripletTweak_Expanded_CDTrue_WUTrue_SamplerFalse_P1N1_ -od ../output/240516_TripletTweaks_IntervalClustering/ -tbcralign ${tbcralign} -tcrdist ${tcrdist} -f ${idf} -model_folder ${model_folder}
