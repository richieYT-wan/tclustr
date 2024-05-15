HOMEDIR=/Users/riwa/Documents/code/tclustr/
PYDIR=${HOMEDIR}pyscripts/
source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda
source activate cuda

# Get the full path of the script
script_path="$(readlink -f "$0")"

# Extract the basename of the script
script_name="$(basename "$script_path" )"

echo "The basename of the script is: $script_name"



cd ${PYDIR}
# Put stuff that don't change at first (np = 500, kf = 0)
# sed ExpModel17_ExpData17peps with Outname == ModelName_InputDFname
# sed ExpData17peps with ExpData17peps == InputDFname
# sed 240426_NewMarginFullExp17Peps/240426_1604_17Peps_KL_1e-2_0200_KFold_0_AllTriplet with actual path
python3 ./240420_VAE_Clustering_intervals.py -np 500 -kf 0 -o CNNVAEModelFull_ExpData17peps -od ../output/240515_IntervalClustering/ExpData17peps/ -tbcralign ../output/240411_ClusteringTests/dist_matrices/2404XX_OUTPUT_tbcralign_distmatrix_140peps_labeled.csv -tcrdist ../output/240411_ClusteringTests/dist_matrices/tcrdist3_distmatrix_140peps_new_labeled.csv -f /Users/riwa/Documents/code/tclustr/data/filtered/240418_nettcr_expanded_20binders_17pep_POSONLY.csv -model_folder ../output/240515_1205_CNNVAE_KF0_GroupSamplerLARGER_KFold_0_XIgoLL -rb True -bf ../output/240515_IntervalClustering -n_jobs 8 -dn ExpData17peps