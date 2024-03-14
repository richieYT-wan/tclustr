source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate cuda

HOMEDIR=/home/projects/vaccine/people/yatwan/tclustr/
PYDIR=${HOMEDIR}pyscripts/
# filename=240110_Bimodal_FullTCR_PepWeightedCosClf_fold_1_BWuPw
cd ${PYDIR}
python3 ./231208_bimodal_vae.py -f /home/projects/vaccine/people/yatwan/tclustr/data/filtered/231205_nettcr_old_26pep_with_swaps.csv -pad -20 -enc BL50LO -cuda True -ne 2000 -lwseq 3 -lwkld 1e-2 -lwvae 1 -lwtrp 1 -lwclf 1 -dist_type cosine -margin 0.075 -mla1 7 -mla2 8 -mlb1 6 -mlb2 7 -mlpep 12 -nl 64 -nh 128 -nhclf 64 -do 0.2 -bn True -n_layers 1 -bs 1024 -lr 5e-4 -wd 1e-4 -wu 15 -wuclf 1000 -pepenc BL50LO -o 240119_2stageFull_PosEncode_AddPep_1kWU_2kEp_LowMarg -rid PEPEP -kf 1 -seed 1 -pepweight False -addpe True
