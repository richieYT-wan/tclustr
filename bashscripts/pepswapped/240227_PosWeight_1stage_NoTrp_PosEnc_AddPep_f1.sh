source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate cuda

HOMEDIR=/home/projects/vaccine/people/yatwan/tclustr/
PYDIR=${HOMEDIR}pyscripts/
cd ${PYDIR}
python3 ./231102_fulltcr_tripletloss.py -f /home/projects/vaccine/people/yatwan/tclustr/data/filtered/231205_nettcr_old_26pep_with_swaps.csv -pad -20 -enc BL50LO -ne 6000 -cuda True -lwseq 1 -lwkld 1e-2 -lwtrp 0 -dist_type cosine -margin 0.075 -mla1 7 -mla2 8 -mla3 22 -mlb1 6 -mlb2 7 -mlb3 23 -mlpep 12 -nl 64 -nh 128 -bs 1024 -lr 5e-4 -wd 3e-4 -wu 15 -o 240227_ONESTAGE_NoTrp_AddPepswap_PosEnc_PosWeight_f1 -rid LMOR1 -kf 1 -seed 1 -addpe True -pepweight True -posweight True #pepweight here is used to do the triplet loss!

