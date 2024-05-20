#! /usr/bin/bash

ms=(Smaller LARGER)
datasource=(78peps ExpAll Old20peps OldAll)
for model_size in "${ms[@]}";do
	for ds in "${datasource[@]}"; do
		# Define the inputs and baselines
		if [ "$ds" = "78peps" -o "$ds" = "ExpAll" ]; then
			input_df=(/home/projects/vaccine/people/yatwan/tclustr/data/filtered/240418_nettcr_expanded_20binders_17pep_POSONLY.csv /home/projects/vaccine/people/yatwan/tclustr/data/filtered/240507_nettcr_exp_pruned_noswap_78peps.csv)
			input_id=(Exp17Peps Exp78Peps)
			tbcralign=/home/projects/vaccine/people/yatwan/tclustr/output/240411_ClusteringTests/dist_matrices/2404XX_OUTPUT_tbcralign_distmatrix_140peps_labeled.csv
			tcrdist=/home/projects/vaccine/people/yatwan/tclustr/output/240411_ClusteringTests/dist_matrices/tcrdist3_distmatrix_140peps_new_labeled.csv

		elif [ "$ds" = "Old20peps" -o "ds" = "OldAll" ]; then
			input_df=(/home/projects/vaccine/people/yatwan/tclustr/data/filtered/240507_nettcr_old_pruned_noswap_20peps.csv /home/projects/vaccine/people/yatwan/tclustr/data/filtered/240416_nettcr_old_26pep_no_swaps.csv /home/projects/vaccine/people/yatwan/tclustr/data/filtered/240416_nettcr_old_top15peps_no_swaps.csv)
			input_id=(Old20peps OldFull Old15Peps)
			tbcralign=/home/projects/vaccine/people/yatwan/tclustr/output/240411_ClusteringTests/dist_matrices/2404XX_OUTPUT_tbcralign_distmatrix_26peps_labeled.csv 
			tcrdist=/home/projects/vaccine/people/yatwan/tclustr/output/240411_ClusteringTests/dist_matrices/tcrdist3_distmatrix_26peps_old_labeled.csv

		fi
		for i in "${!input_df[@]}"; do
			pattern="${datasource}_${model_size}"
			# Define the pattern to match to find the model_folder
			match=$(ls /home/projects/vaccine/people/yatwan/tclustr/output/240516_CNNVAE_DiffDatasets/ | grep ${datasource} | grep ${model_size})
			filename="$(pwd)/clustering_${datasource}_${model_size}_${input_id[i]}.sh"
			model_folder="/home/projects/vaccine/people/yatwan/tclustr/output/240516_CNNVAE_DiffDatasets/${match}"
			script_content=$(cat <<EOF
source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate cuda

HOMEDIR=/home/projects/vaccine/people/yatwan/tclustr/
PYDIR=\${HOMEDIR}pyscripts/
cd \${PYDIR}
model_folder=${model_folder}
tbcralign=${tbcralign}
tcrdist=${tcrdist}
iid=${input_id[i]}
idf=${input_df[i]}
python3 ./240420_VAE_Clustering_intervals.py -np 500 -kf 0 -o \${iid}_CNNVAE_${datasource}_${model_size} -od ../output/240516_TripletTweaks_IntervalClustering/ -tbcralign \${tbcralign} -tcrdist \${tcrdist} -f \${idf} -model_folder \${model_folder} -rb True -n_jobs 40 -dn \${iid} -bf ../output/240515_IntervalClustering
EOF
)
			echo "$script_content" > "${filename}"
			chmod +x ${filename}
		done
	done
done