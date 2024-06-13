# run using a NEW model instance
nohup python3 train_classifier_frozen_mmvae.py -json_file /Users/riwa/Documents/code/tclustr/output/240404_FirstBestLast_comparison/mmvaes/240325_1327_BSSVAE_addlencTrue_addldecTrue_bnTrue_LONG25kepochs_wd1e-6_lwseq2_nhtcr256_nhpep128_nl100_KFold_0_IzcVOH/checkpoint_best_kcv_f00_240325_1327_BSSVAE_addlencTrue_addldecTrue_bnTrue_LONG25kepochs_wd1e-6_lwseq2_nhtcr256_nhpep128_nl100_KFold_0_IzcVOH_JSON_kwargs.json
-pt_file /Users/riwa/Documents/code/tclustr/output/240404_FirstBestLast_comparison/mmvaes/240325_1327_BSSVAE_addlencTrue_addldecTrue_bnTrue_LONG25kepochs_wd1e-6_lwseq2_nhtcr256_nhpep128_nl100_KFold_0_IzcVOH/epoch_1_interval_checkpoint__fold00_kcv_240314_multimodal_NO_HUMAN_tcr_pep_f00_240321_1427_BSSVAE_addlencTrue_addldecTrue_bnTrue_2p5kepochs_wd5e-5_lwseq2_nhtcr256_nhpep128_nl100_KFold_0_VoZuWy.pt
-rid IzcVOH -od RESET_PARAMS_LATENT_TEST/ -o NEW_MODEL -cuda False
-f /Users/riwa/Documents/code/tclustr/data/multimodal/240326_nettcr_paired_withswaps.csv
-nh 64 -do 0.25 -bn True -n_layers 1 -lr 1e-4 -wd 5e-6 -bs 2048 -ne 1500
-pepenc BL50LO -pepweight False -kf 0 -seed 0  -reset True -newmodel True > ../logs/240404_TestFrozenMMVAECLF_NEWMODEL.log 2>&1 &

# run using No RESET but loading first epoch
nohup python3 train_classifier_frozen_mmvae.py -json_file /Users/riwa/Documents/code/tclustr/output/240404_FirstBestLast_comparison/mmvaes/240325_1327_BSSVAE_addlencTrue_addldecTrue_bnTrue_LONG25kepochs_wd1e-6_lwseq2_nhtcr256_nhpep128_nl100_KFold_0_IzcVOH/checkpoint_best_kcv_f00_240325_1327_BSSVAE_addlencTrue_addldecTrue_bnTrue_LONG25kepochs_wd1e-6_lwseq2_nhtcr256_nhpep128_nl100_KFold_0_IzcVOH_JSON_kwargs.json
-pt_file /Users/riwa/Documents/code/tclustr/output/240404_FirstBestLast_comparison/mmvaes/240325_1327_BSSVAE_addlencTrue_addldecTrue_bnTrue_LONG25kepochs_wd1e-6_lwseq2_nhtcr256_nhpep128_nl100_KFold_0_IzcVOH/epoch_1_interval_checkpoint__fold00_kcv_240314_multimodal_NO_HUMAN_tcr_pep_f00_240321_1427_BSSVAE_addlencTrue_addldecTrue_bnTrue_2p5kepochs_wd5e-5_lwseq2_nhtcr256_nhpep128_nl100_KFold_0_VoZuWy.pt
-rid IzcVOH -od RESET_PARAMS_LATENT_TEST/ -o NO_RESET -cuda False
-f /Users/riwa/Documents/code/tclustr/data/multimodal/240326_nettcr_paired_withswaps.csv
-nh 64 -do 0.25 -bn True -n_layers 1 -lr 1e-4 -wd 5e-6 -bs 2048 -ne 1500
-pepenc BL50LO -pepweight False -kf 0 -seed 0 -reset False > ../logs/240404_TestFrozenMMVAECLF_NoReset.log 2>&1 ;

# run using RESET PARAMS
nohup python3 train_classifier_frozen_mmvae.py -json_file /Users/riwa/Documents/code/tclustr/output/240404_FirstBestLast_comparison/mmvaes/240325_1327_BSSVAE_addlencTrue_addldecTrue_bnTrue_LONG25kepochs_wd1e-6_lwseq2_nhtcr256_nhpep128_nl100_KFold_0_IzcVOH/checkpoint_best_kcv_f00_240325_1327_BSSVAE_addlencTrue_addldecTrue_bnTrue_LONG25kepochs_wd1e-6_lwseq2_nhtcr256_nhpep128_nl100_KFold_0_IzcVOH_JSON_kwargs.json
-pt_file /Users/riwa/Documents/code/tclustr/output/240404_FirstBestLast_comparison/mmvaes/240325_1327_BSSVAE_addlencTrue_addldecTrue_bnTrue_LONG25kepochs_wd1e-6_lwseq2_nhtcr256_nhpep128_nl100_KFold_0_IzcVOH/epoch_1_interval_checkpoint__fold00_kcv_240314_multimodal_NO_HUMAN_tcr_pep_f00_240321_1427_BSSVAE_addlencTrue_addldecTrue_bnTrue_2p5kepochs_wd5e-5_lwseq2_nhtcr256_nhpep128_nl100_KFold_0_VoZuWy.pt
-rid IzcVOH -od RESET_PARAMS_LATENT_TEST/ -o RESET_PARAMS -cuda False
-f /Users/riwa/Documents/code/tclustr/data/multimodal/240326_nettcr_paired_withswaps.csv
-nh 64 -do 0.25 -bn True -n_layers 1 -lr 1e-4 -wd 5e-6 -bs 2048 -ne 1500
-pepenc BL50LO -pepweight False -kf 0 -seed 0 -reset True > ../logs/240404_TestFrozenMMVAECLF_RESETONLY.log 2>&1

# run using RANDOM LATENT
nohup python3 train_classifier_frozen_mmvae.py -json_file /Users/riwa/Documents/code/tclustr/output/240404_FirstBestLast_comparison/mmvaes/240325_1327_BSSVAE_addlencTrue_addldecTrue_bnTrue_LONG25kepochs_wd1e-6_lwseq2_nhtcr256_nhpep128_nl100_KFold_0_IzcVOH/checkpoint_best_kcv_f00_240325_1327_BSSVAE_addlencTrue_addldecTrue_bnTrue_LONG25kepochs_wd1e-6_lwseq2_nhtcr256_nhpep128_nl100_KFold_0_IzcVOH_JSON_kwargs.json
-pt_file /Users/riwa/Documents/code/tclustr/output/240404_FirstBestLast_comparison/mmvaes/240325_1327_BSSVAE_addlencTrue_addldecTrue_bnTrue_LONG25kepochs_wd1e-6_lwseq2_nhtcr256_nhpep128_nl100_KFold_0_IzcVOH/epoch_1_interval_checkpoint__fold00_kcv_240314_multimodal_NO_HUMAN_tcr_pep_f00_240321_1427_BSSVAE_addlencTrue_addldecTrue_bnTrue_2p5kepochs_wd5e-5_lwseq2_nhtcr256_nhpep128_nl100_KFold_0_VoZuWy.pt
-rid IzcVOH -od RESET_PARAMS_LATENT_TEST/ -o RANDOM_LATENT -cuda False
-f /Users/riwa/Documents/code/tclustr/data/multimodal/240326_nettcr_paired_withswaps.csv
-nh 64 -do 0.25 -bn True -n_layers 1 -lr 1e-4 -wd 5e-6 -bs 2048 -ne 1500
-pepenc BL50LO -pepweight False -kf 0 -seed 0 -reset False -random_latent True > ../logs/240404_TestFrozenMMVAECLF_RANDOMLATENT.log 2>&1

# Run using RESET + no pep encoding
nohup python3 train_classifier_frozen_mmvae.py -json_file /Users/riwa/Documents/code/tclustr/output/240404_FirstBestLast_comparison/mmvaes/240325_1327_BSSVAE_addlencTrue_addldecTrue_bnTrue_LONG25kepochs_wd1e-6_lwseq2_nhtcr256_nhpep128_nl100_KFold_0_IzcVOH/checkpoint_best_kcv_f00_240325_1327_BSSVAE_addlencTrue_addldecTrue_bnTrue_LONG25kepochs_wd1e-6_lwseq2_nhtcr256_nhpep128_nl100_KFold_0_IzcVOH_JSON_kwargs.json
-pt_file /Users/riwa/Documents/code/tclustr/output/240404_FirstBestLast_comparison/mmvaes/240325_1327_BSSVAE_addlencTrue_addldecTrue_bnTrue_LONG25kepochs_wd1e-6_lwseq2_nhtcr256_nhpep128_nl100_KFold_0_IzcVOH/epoch_1_interval_checkpoint__fold00_kcv_240314_multimodal_NO_HUMAN_tcr_pep_f00_240321_1427_BSSVAE_addlencTrue_addldecTrue_bnTrue_2p5kepochs_wd5e-5_lwseq2_nhtcr256_nhpep128_nl100_KFold_0_VoZuWy.pt
-rid IzcVOH -od RESET_PARAMS_LATENT_TEST/ -o RESET_NOPEPENC -cuda False
-f /Users/riwa/Documents/code/tclustr/data/multimodal/240326_nettcr_paired_withswaps.csv
-nh 64 -do 0.25 -bn True -n_layers 1 -lr 1e-4 -wd 5e-6 -bs 2048 -ne 1500
-pepenc none -pepweight False -kf 0 -seed 0 -reset True -random_latent False > ../logs/240404_TestFrozenMMVAECLF_RESET_NOPEPENC.log 2>&1 &

# Run using df without "original_peptide" column and reset model
nohup python3 train_classifier_frozen_mmvae.py -json_file /Users/riwa/Documents/code/tclustr/output/240404_FirstBestLast_comparison/mmvaes/240325_1327_BSSVAE_addlencTrue_addldecTrue_bnTrue_LONG25kepochs_wd1e-6_lwseq2_nhtcr256_nhpep128_nl100_KFold_0_IzcVOH/checkpoint_best_kcv_f00_240325_1327_BSSVAE_addlencTrue_addldecTrue_bnTrue_LONG25kepochs_wd1e-6_lwseq2_nhtcr256_nhpep128_nl100_KFold_0_IzcVOH_JSON_kwargs.json
-pt_file /Users/riwa/Documents/code/tclustr/output/240404_FirstBestLast_comparison/mmvaes/240325_1327_BSSVAE_addlencTrue_addldecTrue_bnTrue_LONG25kepochs_wd1e-6_lwseq2_nhtcr256_nhpep128_nl100_KFold_0_IzcVOH/epoch_1_interval_checkpoint__fold00_kcv_240314_multimodal_NO_HUMAN_tcr_pep_f00_240321_1427_BSSVAE_addlencTrue_addldecTrue_bnTrue_2p5kepochs_wd5e-5_lwseq2_nhtcr256_nhpep128_nl100_KFold_0_VoZuWy.pt
-rid IzcVOH -od RESET_PARAMS_LATENT_TEST/ -o RESET_NOoriginal -cuda False
-f /Users/riwa/Documents/code/tclustr/data/multimodal/240404_nettcr_swapped_no_original.csv
-nh 64 -do 0.25 -bn True -n_layers 1 -lr 1e-4 -wd 5e-6 -bs 2048 -ne 1500 -pepenc BL50LO -pepweight False -kf 0 -seed 0 -reset True -random_latent False > ../logs/240404_TestFrozenMMVAECLF_RESET_NOoriginal.log 2>&1 &

# RUN Using TCR Blosum only
nohup python3 train_classifier_frozen_mmvae.py -json_file /Users/riwa/Documents/code/tclustr/output/240404_FirstBestLast_comparison/mmvaes/240325_1327_BSSVAE_addlencTrue_addldecTrue_bnTrue_LONG25kepochs_wd1e-6_lwseq2_nhtcr256_nhpep128_nl100_KFold_0_IzcVOH/checkpoint_best_kcv_f00_240325_1327_BSSVAE_addlencTrue_addldecTrue_bnTrue_LONG25kepochs_wd1e-6_lwseq2_nhtcr256_nhpep128_nl100_KFold_0_IzcVOH_JSON_kwargs.json -pt_file /Users/riwa/Documents/code/tclustr/output/240404_FirstBestLast_comparison/mmvaes/240325_1327_BSSVAE_addlencTrue_addldecTrue_bnTrue_LONG25kepochs_wd1e-6_lwseq2_nhtcr256_nhpep128_nl100_KFold_0_IzcVOH/epoch_1_interval_checkpoint__fold00_kcv_240314_multimodal_NO_HUMAN_tcr_pep_f00_240321_1427_BSSVAE_addlencTrue_addldecTrue_bnTrue_2p5kepochs_wd5e-5_lwseq2_nhtcr256_nhpep128_nl100_KFold_0_VoZuWy.pt -rid IzcVOH -od RESET_PARAMS_LATENT_TEST/ -o NEW_MODEL_TCRBLSM -cuda False -f /Users/riwa/Documents/code/tclustr/data/multimodal/240326_nettcr_paired_withswaps.csv -nh 64 -do 0.25 -bn True -n_layers 1 -lr 1e-4 -wd 5e-6 -bs 2048 -ne 1500 -pepenc BL50LO -pepweight False -kf 0 -seed 0 -reset True -newmodel True -tcr_enc BL50LO

# DO ONE MORE RUN : Use a single stage vae with all CDRs and get "reset params" CLF performance
 nohup python3 ./train_classifier_frozen_vae.py -json_file ../output/vaes_models_retrain_old_expdata/240326_1750_240326_1stage_NoTrp_AddAll_ExpData_KFold_0_AAAAAA/checkpoint_best_fold00_kcv_240326_nettcr_paired_withswaps_f00_240326_1750_240326_1stage_NoTrp_AddAll_ExpData_KFold_0_AAAAAA_JSON_kwargs.json -pt_file ../output/vaes_models_retrain_old_expdata/240326_1750_240326_1stage_NoTrp_AddAll_ExpData_KFold_0_AAAAAA/epoch_1000_interval_checkpoint__fold00_kcv_240326_nettcr_paired_withswaps_f00_240326_1750_240326_1stage_NoTrp_AddAll_ExpData_KFold_0_AAAAAA.pt -rid AAAAAA -od RESET_PARAMS_LATENT_TEST/ -o RESET_PARAMS -cuda False -f /Users/riwa/Documents/code/tclustr/data/multimodal/240326_nettcr_paired_withswaps.csv -nh 64 -do 0.25 -bn True -n_layers 1 -lr 1e-4 -wd 5e-6 -bs 2048 -ne 1500 -pepenc BL50LO -pepweight False -kf 0 -seed 0 -reset True > ../logs/240404_TestFrozenOLDVAECLF_RESETONLY.log 2>&1
