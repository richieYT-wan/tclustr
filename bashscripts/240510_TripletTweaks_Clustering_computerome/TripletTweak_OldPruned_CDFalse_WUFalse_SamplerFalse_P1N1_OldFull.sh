source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate cuda

HOMEDIR=/home/projects/vaccine/people/yatwan/tclustr/
PYDIR=${HOMEDIR}pyscripts/
cd ${PYDIR}
model_folder=/home/projects/vaccine/people/yatwan/tclustr/output/240508_TripletTweaks/240514_2312_TripletTweak_OldPruned_CDFalse_WUFalse_SamplerFalse_P1N1_KFold_0_vihkAC
tbcralign=/home/projects/vaccine/people/yatwan/tclustr/output/240411_ClusteringTests/dist_matrices/2404XX_OUTPUT_tbcralign_distmatrix_26peps_labeled.csv
tcrdist=/home/projects/vaccine/people/yatwan/tclustr/output/240411_ClusteringTests/dist_matrices/tcrdist3_distmatrix_26peps_old_labeled.csv
iid=OldFull
idf=/home/projects/vaccine/people/yatwan/tclustr/data/filtered/240416_nettcr_old_26pep_no_swaps.csv
python3 ./240420_VAE_Clustering_intervals.py -np 500 -kf 0 -o TripletTweak_OldPruned_CDFalse_WUFalse_SamplerFalse_P1N1_ -od ../output/240516_TripletTweaks_IntervalClustering/ -tbcralign ${tbcralign} -tcrdist ${tcrdist} -f ${idf} -model_folder ${model_folder}
