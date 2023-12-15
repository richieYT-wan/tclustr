source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate newcuda

HOMEDIR=/home/projects/vaccine/people/yatwan/tclustr/
PYDIR=${HOMEDIR}pyscripts/
filename=TripletTuning_l2_15_32_0.75_1_fold_1_kNP2x
cd ${PYDIR}
python3 ./231102_fulltcr_tripletloss.py -f /home/projects/vaccine/people/yatwan/tclustr/data/filtered/230927_nettcr_positives_only.csv -pad -20 -enc BL50LO -ne 50 -cuda True -lwseq 3 -lwkld 1 -lwtrp 0.75 -dist_type l2 -margin 15 -mla1 0 -mla2 0 -mlb1 0 -mlb2 0 -nl 10 -nh 20 -bs 512 -lr 1.75e-4 -wd 1e-4 -wu 10 -o DELETE_ME -rid DELETE_ME -kf 1 -seed 1 > ~/wtf_cudagain.log
