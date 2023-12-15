source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate cuda

HOMEDIR=/home/projects/vaccine/people/yatwan/tclustr/
PYDIR=${HOMEDIR}pyscripts/
filename=TripletTuning_cosine_0.15_32_0.75_1e-3_fold_3_t5FFN
cd ${PYDIR}
python3 ./231102_fulltcr_tripletloss.py -f /home/projects/vaccine/people/yatwan/tclustr/data/filtered/230927_nettcr_positives_only.csv -pad -20 -enc BL50LO -ne 27500 -cuda True -lwseq 3 -lwkld 1e-3 -lwtrp 0.75 -dist_type cosine -margin 0.15 -mla1 0 -mla2 0 -mlb1 0 -mlb2 0 -nl 32 -nh 64 -bs 512 -lr 1.75e-4 -wd 1e-4 -wu 10 -o TripletTuning_cosine_0.15_32_0.75_1e-3 -rid t5FFN -kf 3 -seed 3
