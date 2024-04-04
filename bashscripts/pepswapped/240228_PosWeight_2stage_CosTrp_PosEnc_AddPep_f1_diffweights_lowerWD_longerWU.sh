source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate cuda

HOMEDIR=/home/projects/vaccine/people/yatwan/tclustr/
PYDIR=${HOMEDIR}pyscripts/
# This script was ran with pos_weights at 1 for cdr1-2-pep and 3 for cdr3s
# and longer wu
cd ${PYDIR}
python3 ./231208_bimodal_vae.py -f /home/projects/vaccine/people/yatwan/tclustr/data/filtered/231205_nettcr_old_26pep_with_swaps.csv -pad -20 -enc BL50LO -cuda True -ne 5000 -lwseq 1 -lwkld 1e-2 -lwvae 1 -lwtrp 1 -lwclf 1 -dist_type cosine -margin 0.075 -mla1 7 -mla2 8 -mlb1 6 -mlb2 7 -mlpep 12 -nl 64 -nh 128 -nhclf 64 -do 0.2 -bn True -n_layers 1 -bs 1024 -lr 5e-4 -wd 1e-5 -wu 30 -wuclf 2000 -pepenc BL50LO -o 240227_TWOSTAGE_CosTrp_AddPepswap_PosEnc_PosWeight_f1 -rid LMOR2 -kf 1 -seed 1 -pepweight True -addpe True -posweight True
