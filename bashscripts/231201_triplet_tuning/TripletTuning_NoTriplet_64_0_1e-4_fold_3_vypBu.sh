source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate cuda

HOMEDIR=/home/projects/vaccine/people/yatwan/tclustr/
PYDIR=${HOMEDIR}pyscripts/
filename=TripletTuning_NoTriplet_64_0_1e-4_fold_3_vypBu
cd ${PYDIR}
python3 ./231102_fulltcr_tripletloss.py -f /home/projects/vaccine/people/yatwan/tclustr/data/filtered/230927_nettcr_positives_only.csv -pad -20 -enc BL50LO -ne 30000 -cuda True -lwseq 3 -lwkld 1e-4 -lwtrp 0 -dist_type 'cosine' -margin None -mla1 0 -mla2 0 -mlb1 0 -mlb2 0 -nl 64 -nh 128 -bs 512 -lr 1.5e-4 -wd 1e-4 -wu 10 -o TripletTuning_NoTriplet_64_0_1e-4 -rid vypBu -kf 3 -seed 3
