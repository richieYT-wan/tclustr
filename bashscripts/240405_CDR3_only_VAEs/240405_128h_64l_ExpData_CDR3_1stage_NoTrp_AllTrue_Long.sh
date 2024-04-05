source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate cuda

HOMEDIR=/home/projects/vaccine/people/yatwan/tclustr/
PYDIR=${HOMEDIR}pyscripts/
cd ${PYDIR}
python3 ./231102_fulltcr_tripletloss.py -f /home/projects/vaccine/people/yatwan/tclustr/data/multimodal/240311_nettcr_pairedAB_expanded_noswap.csv -pad -20 -enc BL50LO -ne 10000 -cuda True -lwseq 1 -lwkld 1e-2 -lwtrp 0 -dist_type cosine -margin 0.075 -mla1 0 -mla2 0 -mla3 22 -mlb1 0 -mlb2 0 -mlb3 23 -mlpep 0 -nl 64 -nh 128 -bs 1024 -lr 5e-4 -wd 1e-5 -wu 100 -o ExpData_CDR3ONLY_1stage_SMALL_128h_64l_NoTrp -rid smallTCRP1 -kf 0 -seed 0 -addpe True -bn True -ale True -ald True -ob False -pepweight True -posweight True > "${HOMEDIR}logs/240405_CDR3ONLY_1stage_small.log" 2>&1
