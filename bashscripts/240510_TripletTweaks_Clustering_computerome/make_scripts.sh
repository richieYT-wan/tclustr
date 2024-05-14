#! /usr/bin/bash

# 140 peps
-tbcralign ../output/240411_ClusteringTests/dist_matrices/2404XX_OUTPUT_tbcralign_distmatrix_140peps_labeled.csv 
-tcrdist ../output/240411_ClusteringTests/dist_matrices/tcrdist3_distmatrix_140peps_new_labeled.csv

# 26 peps
tbcralign=../output/240411_ClusteringTests/dist_matrices/2404XX_OUTPUT_tbcralign_distmatrix_26peps_labeled.csv 
tcrdist=../output/240411_ClusteringTests/dist_matrices/tcrdist3_distmatrix_26peps_old_labeled.csv

sampler=(True False)
warmup=(True False)
cooldown=(True False)
tripletweight=(P1N1 P1N0 P0N1)
datasource=(Expanded OldPruned)
for samp in "${sampler[@]}";do
	for wu in "${warmup[@]}"; do
		for cd in "${cooldown[@]}"; do
			for tw in "${tripletweight[@]}"; do
				for ds in "${datasource[@]}"; do
					# Define the inputs and baselines
					if [ "$ds" = "Expanded" ]; then
						input_df=(/home/projects/vaccine/people/yatwan/tclustr/data/filtered/240418_nettcr_expanded_20binders_17pep_POSONLY.csv /home/projects/vaccine/people/yatwan/tclustr/data/filtered/240507_nettcr_exp_pruned_noswap_78peps.csv)
						tbcralign=/home/projects/vaccine/people/yatwan/tclustr/output/240411_ClusteringTests/dist_matrices/2404XX_OUTPUT_tbcralign_distmatrix_140peps_labeled.csv
						tcrdist=/home/projects/vaccine/people/yatwan/tclustr/output/240411_ClusteringTests/dist_matrices/tcrdist3_distmatrix_140peps_new_labeled.csv

					elif [ "$ds" = "OldPruned" ]; then
						input_df=(/home/projects/vaccine/people/yatwan/tclustr/data/filtered/240507_nettcr_old_pruned_noswap_20peps.csv /home/projects/vaccine/people/yatwan/tclustr/data/filtered/240416_nettcr_old_26pep_no_swaps.csv /home/projects/vaccine/people/yatwan/tclustr/data/filtered/240416_nettcr_old_top15peps_no_swaps.csv)
						tbcralign=/home/projects/vaccine/people/yatwan/tclustr/output/240411_ClusteringTests/dist_matrices/2404XX_OUTPUT_tbcralign_distmatrix_26peps_labeled.csv 
						tcrdist=/home/projects/vaccine/people/yatwan/tclustr/output/240411_ClusteringTests/dist_matrices/tcrdist3_distmatrix_26peps_old_labeled.csv

					fi
					# Define the pattern to match to find the model_folder
					pattern="TripletTweak_${ds}_CD${cd}_WU${wu}_Sampler${samp}_${tw}"
					match=$(ls /home/projects/vaccine/people/yatwan/tclustr/output/240508_TripletTweaks/ | grep ${pattern})
					model_folder="/home/projects/vaccine/people/yatwan/tclustr/output/240508_TripletTweaks/${match}"
					echo $match
					# echo "${pattern}xxxxXXXXxxxx${tbcralign}xxxxXXXXxxxx${tcrdist}"
					# echo "${input_df[@]}"

				done
			done
		done
	done
done