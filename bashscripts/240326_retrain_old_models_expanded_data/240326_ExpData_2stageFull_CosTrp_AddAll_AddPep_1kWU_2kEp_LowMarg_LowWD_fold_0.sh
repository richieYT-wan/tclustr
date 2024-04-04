source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate cuda

HOMEDIR=/home/projects/vaccine/people/yatwan/tclustr/
PYDIR=${HOMEDIR}pyscripts/
# filename=240110_Bimodal_FullTCR_PepWeightedCosClf_fold_1_BWuPw
# Using pepweights to set 0-1 on triplet loss
cd ${PYDIR}
python3 ./231208_bimodal_vae.py -f /home/projects/vaccine/people/yatwan/tclustr/data/multimodal/240326_nettcr_paired_withswaps.csv -pad -20 -enc BL50LO -cuda True -ne 4000 -lwseq 1 -lwkld 1e-2 -lwvae 1 -lwtrp 1 -lwclf 1 -dist_type cosine -margin 0.075 -mla1 7 -mla2 8 -mlb1 6 -mlb2 7 -mlpep 12 -nl 100 -nh 256 -nhclf 64 -do 0.2  -n_layers 1 -bs 1024 -lr 5e-4 -wd 1e-5 -wu 50 -wuclf 1250 -pepenc BL50LO -o 240326_2stage_CosTrp_AddAll_ExpData -rid DDDDDD -kf 0 -seed 0 -posweight True -pepweight True -addpe False -bn True -ale True -ald True -ob False
