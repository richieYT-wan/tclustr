source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate cuda

HOMEDIR=/home/projects/vaccine/people/yatwan/tclustr/
PYDIR=${HOMEDIR}pyscripts/
# filename=240110_Bimodal_FullTCR_PepWeightedCosClf_fold_1_BWuPw
cd ${PYDIR}
python3 ./231208_bimodal_vae.py -f /Users/riwa/Documents/code/tclustr/data/filtered/231205_nettcr_old_26pep_with_swaps.csv -pad -20 -enc BL50LO -cuda True -ne 2000 -lwseq 3 -lwkld 1e-2 -lwvae 1 -lwtrp 1 -lwclf 1 -dist_type cosine -margin 0.075 -mla1 7 -mla2 8 -mlb1 6 -mlb2 7 -nl 64 -nh 128 -nhclf 64 -do 0.2 -bn True -n_layers 1 -bs 1024 -lr 5e-4 -wd 1e-4 -wu 15 -wuclf 1000 -pepenc BL50LO -o 240115_twostageFullTCR_WarmUpLowerMargin2Kepochs -rid TIGHT -kf 1 -seed 1 -pepweight False
