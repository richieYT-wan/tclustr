#! /usr/bin/bash
HOMEDIR=/home/projects/vaccine/people/yatwan/tclustr/
PYDIR=${HOMEDIR}pyscripts/

source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate cuda

# Get the full path of the script
script_path="$(readlink -f "$0")"

# Extract the basename of the script
script_name="$(basename "$script_path" .sh)"

echo "The basename of the script is: $script_name"



cd ${PYDIR}
# Put stuff that don't change at first (np = 500, kf = 0)
python3 ./240420_VAE_Clustering_intervals.py -np 500 -kf 0 -o OutnameHere -od ../output/240515_IntervalClustering/OutdirHere/ -tbcralign ../output/240411_ClusteringTests/dist_matrices/2404XX_OUTPUT_tbcralign_distmatrix_26peps_labeled.csv -tcrdist ../output/output/240411_ClusteringTests/dist_matrices/tcrdist3_distmatrix_26peps_old_labeled.csv -f /home/projects/vaccine/people/yatwan/tclustr/data/filtered/240507_nettcr_old_pruned_noswap_20peps.csv -model_folder ../output/240514_ModelsForClustering/OtherFolder/ModelFolderName