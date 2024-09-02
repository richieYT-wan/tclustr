for i in $(seq -f "%04g" 1 45); 
do 
filename="mixed_${i}"
script_content=$(cat <<EOF
#! /usr/bin/bash

cd /home/projects/vaccine/people/yatwan/tclustr/bashscripts/
sh MSTcut_all_pipeline.sh -f ../data/OTS/mixed_covid_healthy/${filename}.txt -c A1 A2 A3 B1 B2 B3 -s c2 -l Disease -e Disease Source count index_col Run -i index_col -o QSUB_MST_mixed_${i} -p ../output/240618_NestedKCV_CNNVAE/Nested_TwoStageCNNVAE_latent_128_kld_1e-2_ExpData_KFold_0_240618_1608_pDQhj/checkpoint_best_kcv_fold_00_Nested_TwoStageCNNVAE_latent_128_kld_1e-2_ExpData_KFold_0_240618_1608_pDQhj.pt -j ../output/240618_NestedKCV_CNNVAE/Nested_TwoStageCNNVAE_latent_128_kld_1e-2_ExpData_KFold_0_240618_1608_pDQhj/checkpoint_best_kcv_fold_00_Nested_TwoStageCNNVAE_latent_128_kld_1e-2_ExpData_KFold_0_240618_1608_pDQhj_JSON_kwargs.json
EOF
)
	echo "$script_content" > "./${filename}.sh"
	chmod +x "./${filename}.sh"
done
