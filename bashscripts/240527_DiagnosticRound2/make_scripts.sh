#! /usr/bin/bash

SCRIPTDIR=$(pwd)
trainsource=(ExpAll OldAll)
trainfile=(240326_nettcr_paired_NOswaps.csv 240416_nettcr_old_26pep_no_swaps.csv)
model_type=(VAE CNNVAE)
latdims=(50 100)
klds=(1e-1 1e-2 1e-3)

for model in ${model_type[@]};do
	if [ "$model" = "CNNVAE" ]; then
		python_script=240515_cnnvae_tripletloss.py
	else
		python_script=231102_fulltcr_tripletloss.py
	fi
	for ld in ${ms[@]};do
		for kl in ${klds[@]};do
			for ts in "${!trainsource[@]}"; do
				ds=${trainsource[ts]}
				tf=${trainfile[ts]}
				# TODO HERE WHEN YOU COME BACK
				# FOR ALL THESE CONDITIONS: RUN A SINGLE VAE
				# RUN ALL THE SUB-FILTERING DATASETS FOR THE CLUSTERING PARTS
				# USE N_JOBS 40 ETC
				# USE THE RANDOM ID TO GREP THE CORRECT FOLDER FOR THE INPUT OF THE MODEL
				
				filename="RedoHP_${model}_latent_${ld}_kld_${kl}_train_${ds}"

				script_content=$(cat <<EOF
source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate cuda

HOMEDIR=/home/projects/vaccine/people/yatwan/tclustr/
outdir="240527_DiagnosticRound2"
RESDIR="\${HOMEDIR}output/\${outdir}"
PYDIR=\${HOMEDIR}pyscripts/
cd \${PYDIR}

# Define the characters that can be used
characters="abcdefghijkmnopqrstuvwxyzABCDEFGHJKLMNOPQRSTUVWXYZ0123456789"
# Generate a random index between 0 and 61 (total number of characters)
index=\$((RANDOM % 60))
# Get the character at the generated index
first_char="\${characters:index:1}"
# Generate the remaining 4 characters as a combination of the defined characters
rest_chars=\$(head /dev/urandom | tr -dc "\$characters" | head -c 4)
# Combine the first and remaining characters
random_id="\${first_char}\${rest_chars}"
outname=${filename}

# Run VAE
python3 ./${python_script} -f \${HOMEDIR}data/filtered/${tf} -od \${outdir} -pad -20 -enc BL50LO -ne 20000 -cuda True -lwseq 1 -lwkld ${kl} -lwtrp 3 -dist_type cosine -margin 0.2 -mla1 7 -mla2 8 -mla3 22 -mlb1 6 -mlb2 7 -mlb3 23 -mlpep 0 -nl ${ld} -nh 128 -bs 512 -lr 1e-4 -wd 1e-4 -wu 150 -fp 50 -kld_dec 1e-2 -kldts 0.075 -o \${outname} -kf 0 -seed 0 -addpe True -bn True -ale True -ald True -ob False -pepweight False -posweight True -rid \${random_id}
EOF
)
				echo "$script_content" > "${SCRIPTDIR}${filename}".sh
				# After writing the first script_content to the file
				# use a loop to go over input_df tbcr_align etc and append to the script using >>
				if [ "$ds" = "78peps" -o "$ds" = "ExpAll" ]; then
					input_df=(/home/projects/vaccine/people/yatwan/tclustr/data/filtered/240418_nettcr_expanded_20binders_17pep_POSONLY.csv /home/projects/vaccine/people/yatwan/tclustr/data/filtered/240507_nettcr_exp_pruned_noswap_78peps.csv)
					input_id=(ExpData17peps ExpDataTop78)
					tbcralign=/home/projects/vaccine/people/yatwan/tclustr/output/240411_ClusteringTests/dist_matrices/2404XX_OUTPUT_tbcralign_distmatrix_140peps_labeled.csv
					tcrdist=/home/projects/vaccine/people/yatwan/tclustr/output/240411_ClusteringTests/dist_matrices/tcrdist3_distmatrix_140peps_new_labeled.csv

				elif [ "$ds" = "Old20peps" -o "ds" = "OldAll" ]; then
					input_df=(/home/projects/vaccine/people/yatwan/tclustr/data/filtered/240507_nettcr_old_pruned_noswap_20peps.csv /home/projects/vaccine/people/yatwan/tclustr/data/filtered/240416_nettcr_old_26pep_no_swaps.csv)
					input_id=(OldDataTop20 OldDataNoPrune)
					tbcralign=/home/projects/vaccine/people/yatwan/tclustr/output/240411_ClusteringTests/dist_matrices/2404XX_OUTPUT_tbcralign_distmatrix_26peps_labeled.csv 
					tcrdist=/home/projects/vaccine/people/yatwan/tclustr/output/240411_ClusteringTests/dist_matrices/tcrdist3_distmatrix_26peps_old_labeled.csv

				fi
				for idx in ${!input_id[@]};do
					cluster_content=$(cat <<EOF


outmatch=\$(ls -t \${RESDIR} | grep \${random_id} | head -n 1)
iid=${input_id[i]}
idf=${input_df[i]}
# Run clustering part
python3 ./240420_VAE_Clustering_intervals.py -rid \${random_id} -np 500 -kf 0 -o \${outname}_${iid} -od ../output/\${outdir}/clustering/ -tbcralign ${tbcralign} -tcrdist ${tcrdist} -f ${idf} -model_folder "\${RESDIR}/\${outmatch}" -rb True -n_jobs 40 -dn ${iid} -bf ../output/240515_IntervalClustering

EOF
)
					echo "${cluster_content}$" >> "${SCRIPTDIR}${filename}".sh
				chmod +x ${filename}.sh
				done
			done
		done
	done
done