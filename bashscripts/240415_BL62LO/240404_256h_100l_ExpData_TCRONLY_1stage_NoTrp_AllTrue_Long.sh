source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate cuda

HOMEDIR=/home/projects/vaccine/people/yatwan/tclustr/
PYDIR=${HOMEDIR}pyscripts/
cd ${PYDIR}
python3 ./231102_fulltcr_tripletloss.py -f /home/projects/vaccine/people/yatwan/tclustr/data/multimodal/240311_nettcr_pairedAB_expanded_noswap.csv -pad -20 -enc BL62LO -ne 12000 -cuda True -lwseq 1 -lwkld 1e-2 -lwtrp 0 -dist_type cosine -margin 0.075 -mla1 7 -mla2 8 -mla3 22 -mlb1 6 -mlb2 7 -mlb3 23 -mlpep 0 -nl 100 -nh 256 -bs 1024 -lr 5e-4 -wd 1e-5 -wu 100 -o BL62LO_ExpData_TCRONLY_1stage_LARGE_256h_100l_NoTrp -rid LN62 -kf 0 -seed 0 -addpe True -bn True -ale True -ald True -ob False -pepweight True -posweight True > "${HOMEDIR}logs/BL62LO_240404_1stage_LARGE.log" 2>&1
