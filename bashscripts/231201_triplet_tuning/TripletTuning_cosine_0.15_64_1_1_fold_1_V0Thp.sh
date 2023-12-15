source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate cuda

HOMEDIR=/home/projects/vaccine/people/yatwan/tclustr/
PYDIR=${HOMEDIR}pyscripts/
filename=TripletTuning_cosine_0.15_64_1_1_fold_1_V0Thp
cd ${PYDIR}
python3 ./231102_fulltcr_tripletloss.py -f /home/projects/vaccine/people/yatwan/tclustr/data/filtered/230927_nettcr_positives_only.csv -pad -20 -enc BL50LO -ne 27500 -cuda True -lwseq 3 -lwkld 1 -lwtrp 1 -dist_type cosine -margin 0.15 -mla1 0 -mla2 0 -mlb1 0 -mlb2 0 -nl 64 -nh 128 -bs 512 -lr 1.75e-4 -wd 1e-4 -wu 10 -o TripletTuning_cosine_0.15_64_1_1 -rid V0Thp -kf 1 -seed 1
