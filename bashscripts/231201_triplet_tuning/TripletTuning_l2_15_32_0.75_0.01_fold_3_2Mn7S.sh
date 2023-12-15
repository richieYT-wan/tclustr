source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate cuda

HOMEDIR=/home/projects/vaccine/people/yatwan/tclustr/
PYDIR=${HOMEDIR}pyscripts/
filename=TripletTuning_l2_15_32_0.75_0.01_fold_3_2Mn7S
cd ${PYDIR}
python3 ./231102_fulltcr_tripletloss.py -f /home/projects/vaccine/people/yatwan/tclustr/data/filtered/230927_nettcr_positives_only.csv -pad -20 -enc BL50LO -ne 27500 -cuda True -lwseq 3 -lwkld 0.01 -lwtrp 0.75 -dist_type l2 -margin 15 -mla1 0 -mla2 0 -mlb1 0 -mlb2 0 -nl 32 -nh 64 -bs 512 -lr 1.75e-4 -wd 1e-4 -wu 10 -o TripletTuning_l2_15_32_0.75_0.01 -rid 2Mn7S -kf 3 -seed 3
