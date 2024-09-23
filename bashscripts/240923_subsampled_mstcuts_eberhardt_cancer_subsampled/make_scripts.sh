for number in 7 15 34 51;
do
  for seed in $(seq -f "%03g" 0 99);
  do

    filename="subsampled_cancer_hpv${number}_seed_${seed}"
script_content=$(cat <<EOF
#! /usr/bin/bash

#cd /home/projects/vaccine/people/yatwan/tclustr/bashscripts/
sh /home/projects/vaccine/people/yatwan/tclustr/bashscripts/MSTcut_all_pipeline_4vae_subsampling_hardcoded_models.sh -f "/home/projects/vaccine/people/yatwan/tclustr/data/OTS/eberhardt_garner_mixed_subsampled/hpv${number}_garner_subsampled_seed_${seed}.txt" -c A1 A2 A3 B1 B2 B3 -s c2 -l TSubtype -e TSubtype Disease Source count index_col Run -i index_col -o ${filename}
EOF
)
	echo "$script_content" > "./${filename}.sh"
	chmod +x "./${filename}.sh"
  done
done
