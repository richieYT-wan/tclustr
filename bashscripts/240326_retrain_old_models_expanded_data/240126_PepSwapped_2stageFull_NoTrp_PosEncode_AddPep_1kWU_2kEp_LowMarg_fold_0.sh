source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate cuda

HOMEDIR=/home/projects/vaccine/people/yatwan/tclustr/
PYDIR=${HOMEDIR}pyscripts/
# filename=240110_Bimodal_FullTCR_PepWeightedCosClf_fold_1_BWuPw
# Using pepweights to set 0-1 on triplet loss
cd ${PYDIR}
python3 ./231208_bimodal_vae.py -f /home/projects/vaccine/people/yatwan/tclustr/data/filtered/231205_nettcr2-2_alpha_beta_paired_expanded.csv -pad -20 -enc BL50LO -cuda True -ne 6000 -lwseq 1 -lwkld 1e-2 -lwvae 1 -lwtrp 0 -lwclf 1 -dist_type cosine -margin 0.075 -mla1 7 -mla2 8 -mlb1 6 -mlb2 7 -mlpep 12 -nl 64 -nh 128 -nhclf 64 -do 0.2 -n_layers 1 -bs 2048 -lr 1e-4 -wd 1e-5 -wu 50 -wuclf 1250 -pepenc BL50LO -o 240326_2stage_NoTrp_AddAll_ExpData -rid CCCCC -kf 0 -seed 0 -posweight True -pepweight True -addpe False -bn True -ale True -ald True -ob False
