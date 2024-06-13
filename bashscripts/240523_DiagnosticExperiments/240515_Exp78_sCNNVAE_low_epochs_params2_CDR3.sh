source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate cuda

# Setting input directory paths
HOMEDIR=/home/projects/vaccine/people/yatwan/tclustr/
PYDIR=${HOMEDIR}pyscripts/

# Setting file paths and clustering stuff
file=/home/projects/vaccine/people/yatwan/tclustr/data/filtered/240507_nettcr_exp_pruned_noswap_78peps.csv
tbcralign=/home/projects/vaccine/people/yatwan/tclustr/output/240411_ClusteringTests/dist_matrices/2404XX_OUTPUT_tbcralign_distmatrix_140peps_labeled.csv
tcrdist=/home/projects/vaccine/people/yatwan/tclustr/output/240411_ClusteringTests/dist_matrices/tcrdist3_distmatrix_140peps_new_labeled.csv
iid=ExpDataTop78

# setting output dirs
outname="CDR3_CNNVAE_sm_lowEpochs_params2_overfitted_${iid}"
outdir='240523_DiagnogsticExperiments'
RESDIR="${HOMEDIR}output/${outdir}/"

cd ${PYDIR}
# Run CNNVAE
# python3 ./240515_cnnvae_tripletloss.py -od ${outdir} -f ${file} -pad -20 -enc BL50LO -ne 20000 -cuda True -lwseq 1 -lwkld 1e-2 -lwtrp 10 -dist_type cosine -margin 0.2 -mla1 0 -mla2 0 -mla3 22 -mlb1 0 -mlb2 0 -mlb3 23 -mlpep 0 -op_trans_1 2 -nl 100 -nh 128 -bs 512 -lr 2.5e-4 -wd 1e-4 -wu 150 -fp 150 -kld_dec 1e-3 -kldts 0.075 -o ${outname} -kf 0 -seed 0 -addpe True -bn True -pepweight False -posweight True -minority_sampler True -minority_count 50 

outmatch=$(ls -t ${RESDIR} | grep ${outname} | grep CDR3 | head -n 1)
# Run Clustering
python3 ./240420_VAE_Clustering_intervals.py -np 500 -kf 0 -o ${outname}_${iid} -od ../output/${outdir}/clustering/ -tbcralign ${tbcralign} -tcrdist ${tcrdist} -f ${file} -model_folder "${RESDIR}/${outmatch}" -rb True -n_jobs 40 -dn CDR3_${iid} -bf ../output/240515_IntervalClustering
