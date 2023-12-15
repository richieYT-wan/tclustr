source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate cuda

HOMEDIR=/home/projects/vaccine/people/yatwan/tclustr/
PYDIR=${HOMEDIR}pyscripts/
filename=TripletTuning_l2_5_64_0.75_0.1_fold_4_IHc02
cd ${PYDIR}
python3 ./231102_fulltcr_tripletloss.py -f /home/projects/vaccine/people/yatwan/tclustr/data/filtered/230927_nettcr_positives_only.csv -pad -20 -enc BL50LO -ne 27500 -cuda True -lwseq 3 -lwkld 0.1 -lwtrp 0.75 -dist_type l2 -margin 5 -mla1 0 -mla2 0 -mlb1 0 -mlb2 0 -nl 64 -nh 128 -bs 512 -lr 1.75e-4 -wd 1e-4 -wu 10 -o TripletTuning_l2_5_64_0.75_0.1 -rid IHc02 -kf 4 -seed 4
