source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate cuda

HOMEDIR=/home/projects/vaccine/people/yatwan/tclustr/
PYDIR=${HOMEDIR}pyscripts/
cd ${PYDIR}
model_folder=/home/projects/vaccine/people/yatwan/tclustr/output/240508_TripletTweaks/240514_2312_TripletTweak_OldPruned_CDFalse_WUTrue_SamplerFalse_P1N0_KFold_0_RdN1pa
tbcralign=/home/projects/vaccine/people/yatwan/tclustr/output/240411_ClusteringTests/dist_matrices/2404XX_OUTPUT_tbcralign_distmatrix_26peps_labeled.csv
tcrdist=/home/projects/vaccine/people/yatwan/tclustr/output/240411_ClusteringTests/dist_matrices/tcrdist3_distmatrix_26peps_old_labeled.csv
iid=Old20peps
idf=/home/projects/vaccine/people/yatwan/tclustr/data/filtered/240507_nettcr_old_pruned_noswap_20peps.csv
python3 ./240420_VAE_Clustering_intervals.py -np 400 -kf 0 -o ${iid}_TripletTweak_OldPruned_CDFalse_WUTrue_SamplerFalse_P1N0 -od ../output/240516_TripletTweaks_IntervalClustering/${iid} -tbcralign ${tbcralign} -tcrdist ${tcrdist} -f ${idf} -n_jobs 40 -dn OldDataTop20 -rb True -bf ../output/240515_IntervalClustering/ -model_folder ${model_folder}
