# First: Do a run with a "new model" with a VAE using CDR3s only (no pep)
# Do a new model run using both a Normal VAE and a BSSVAE
# compare with run with normal VAE with all CDRs as well (see 240404_all_runs_logged last command)

# Do 2 "blank" runs of CDR3 VAEs and BSSVAEs loading the first epoch (See 240405_CDR3_only_VAEs and save the first epoch only so I can already pre-run the analysis)
# Do 2 blank runs of CDR3 VAEs and BSSVAEs loading then resetting the weights
HDIR="/Users/riwa/Documents/code/tclustr/"
MDIR="${HDIR}output/240405_CDR3_VAEs/"
df="/Users/riwa/Documents/code/tclustr/data/multimodal/240326_nettcr_paired_withswaps.csv"
outdir="${HDIR}output/240404_RESET_PARAMS_LATENT_TEST/"
# CDR3 old VAE runs : 1 blank (load checkpoint_1)
#                     1 reset (load + reset
cd "${HDIR}pyscripts/"
# At least I'm sure that the TCR blsm enc works for mmvae dataset since I debugged it
nohup python3 train_classifier_frozen_mmvae.py -json_file "${MDIR}CDR3_BSSVAE_frVvuv/checkpoint_json.json" -pt_file "${MDIR}CDR3_BSSVAE_frVvuv/checkpoint_1.pt" -rid frVvuv -od ${outdir} -o CDR3_BLSMonly_noLatent -cuda False -f ${df} -nh 64 -do 0.25 -bn True -n_layers 1 -lr 1e-4 -wd 5e-6 -bs 2048 -ne 1500 -pepenc BL50LO -pepweight False -kf 0 -seed 0 -reset True -tcr_enc BL50LO > ${HDIR}/logs/240405_CDR3_BLOSUM.log 2>&1