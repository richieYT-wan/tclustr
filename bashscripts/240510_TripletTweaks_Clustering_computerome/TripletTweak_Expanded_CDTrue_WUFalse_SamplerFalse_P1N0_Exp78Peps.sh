source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate cuda

HOMEDIR=/home/projects/vaccine/people/yatwan/tclustr/
PYDIR=${HOMEDIR}pyscripts/
cd ${PYDIR}
model_folder=/home/projects/vaccine/people/yatwan/tclustr/output/240508_TripletTweaks/240509_1205_TripletTweak_Expanded_CDTrue_WUFalse_SamplerFalse_P1N0_KFold_0_HDA23H
tbcralign=/home/projects/vaccine/people/yatwan/tclustr/output/240411_ClusteringTests/dist_matrices/2404XX_OUTPUT_tbcralign_distmatrix_140peps_labeled.csv
tcrdist=/home/projects/vaccine/people/yatwan/tclustr/output/240411_ClusteringTests/dist_matrices/tcrdist3_distmatrix_140peps_new_labeled.csv
iid=Exp78Peps
idf=/home/projects/vaccine/people/yatwan/tclustr/data/filtered/240507_nettcr_exp_pruned_noswap_78peps.csv
python3 ./240420_VAE_Clustering_intervals.py -np 400 -kf 0 -o ${iid}_TripletTweak_Expanded_CDTrue_WUFalse_SamplerFalse_P1N0 -od ../output/240516_TripletTweaks_IntervalClustering/${iid} -tbcralign ${tbcralign} -tcrdist ${tcrdist} -f ${idf} -n_jobs 40 -dn ExpDataTop78 -rb True -bf ../output/240515_IntervalClustering/ -model_folder ${model_folder}
