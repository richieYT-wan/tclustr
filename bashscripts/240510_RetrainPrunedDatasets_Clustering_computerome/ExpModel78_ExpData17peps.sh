 
HOMEDIR=/home/projects/vaccine/people/yatwan/tclustr/
PYDIR=${HOMEDIR}pyscripts/

source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate cuda

# Get the full path of the script
script_path="$(readlink -f "$0")"

# Extract the basename of the script
script_name="$(basename "$script_path" )"

echo "The basename of the script is: $script_name"



cd ${PYDIR}
# Put stuff that don't change at first (np = 500, kf = 0)
# sed ExpModel78_ExpData17peps with Outname == ModelName_InputDFname
# sed ExpData17peps with ExpData17peps == InputDFname
# sed 240507_RetrainPrunedDatasets/240507_1253_RetrainPruned_ExpData_78peps_KFold_0_oITlA2 with actual path
python3 ./240420_VAE_Clustering_intervals.py -np 500 -kf 0 -o ExpModel78_ExpData17peps -od ../output/240515_IntervalClustering/ExpData17peps/ -tbcralign ../output/240411_ClusteringTests/dist_matrices/2404XX_OUTPUT_tbcralign_distmatrix_140peps_labeled.csv -tcrdist ../output/240411_ClusteringTests/dist_matrices/tcrdist3_distmatrix_140peps_new_labeled.csv -f /home/projects/vaccine/people/yatwan/tclustr/data/filtered/240418_nettcr_expanded_20binders_17pep_POSONLY.csv -model_folder ../output/240514_ModelsForClustering/240507_RetrainPrunedDatasets/240507_1253_RetrainPruned_ExpData_78peps_KFold_0_oITlA2