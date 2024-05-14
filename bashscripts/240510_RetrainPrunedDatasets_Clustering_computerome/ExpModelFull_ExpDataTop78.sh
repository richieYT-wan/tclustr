#! /usr/bin/b
HOMEDIR=/home/projects/vaccine/people/yatwan/tclustr/
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
# sed ExpModelFull_ExpDataTop78 with Outname == ModelName_InputDFname
# sed ExpDataTop78 with ExpDataTop78 == InputDFname
# sed 240426_NewMarginFullExp17Peps/240426_1612_FullExp_KL_1e-2_0200_KFold_0_AllTriplet with actual path
python3 ./240420_VAE_Clustering_intervals.py -np 500 -kf 0 -o ExpModelFull_ExpDataTop78 -od ../output/240515_IntervalClustering/ExpDataTop78/ -tbcralign ../output/240411_ClusteringTests/dist_matrices/2404XX_OUTPUT_tbcralign_distmatrix_140peps_labeled.csv -tcrdist ../output/240411_ClusteringTests/dist_matrices/tcrdist3_distmatrix_140peps_new_labeled.csv -f /home/projects/vaccine/people/yatwan/tclustr/data/filtered/240507_nettcr_exp_pruned_noswap_78peps.csv -model_folder ../output/240514_ModelsForClustering/240426_NewMarginFullExp17Peps/240426_1612_FullExp_KL_1e-2_0200_KFold_0_AllTriplet