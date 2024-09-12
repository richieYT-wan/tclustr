for number in $(seq -f "%03g" 0 42);
do
  for seed in $(seq -f "%03g" 0 99);
  do

    filename="subsampled_francis_${number}_seed_${seed}"
script_content=$(cat <<EOF
#! /usr/bin/bash

#cd /home/projects/vaccine/people/yatwan/tclustr/bashscripts/
sh /home/projects/vaccine/people/yatwan/tclustr/bashscripts/MSTcut_all_pipeline_4vae_subsampling_hardcoded_refined_models.sh -f ../data/OTS/subsampled_francis_garner/${filename}.txt -c A1 A2 A3 B1 B2 B3 -s c2 -l Disease -e Disease Source count norm_count index_col Run -i index_col -o "${filename}"
EOF
)
	echo "$script_content" > "./${filename}.sh"
	chmod +x "./${filename}.sh"
  done
done
