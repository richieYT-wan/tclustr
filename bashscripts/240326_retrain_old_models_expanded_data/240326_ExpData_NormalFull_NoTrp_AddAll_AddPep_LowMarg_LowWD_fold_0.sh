source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate cuda

HOMEDIR=/home/projects/vaccine/people/yatwan/tclustr/
PYDIR=${HOMEDIR}pyscripts/
cd ${PYDIR}
python3 ./231102_fulltcr_tripletloss.py -f /home/projects/vaccine/people/yatwan/tclustr/data/multimodal/240326_nettcr_paired_withswaps.csv -pad -20 -enc BL50LO -ne 15000 -cuda True -lwseq 1 -lwkld 1e-2 -lwtrp 0 -dist_type cosine -margin 0.075 -mla1 7 -mla2 8 -mla3 22 -mlb1 6 -mlb2 7 -mlb3 23 -mlpep 12 -nl 100 -nh 256 -bs 1024 -lr 1e-4 -wd 1e-5 -wu 50 -o 240326_1stage_NoTrp_AddAll_ExpData -rid AAAAAA -kf 0 -seed 0 -addpe False -bn True -ale True -ald True -ob False -pepweight True -posweight True
