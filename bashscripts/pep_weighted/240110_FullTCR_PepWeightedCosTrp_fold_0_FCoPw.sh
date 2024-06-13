HOMEDIR=/Users/riwa/Documents/code/tclustr/
PYDIR=${HOMEDIR}pyscripts/
filename=240110_FullTCR_PepWeightedCosTrp_fold_0_FCoPw
cd ${PYDIR}
python3 ./231102_fulltcr_tripletloss.py -f /Users/riwa/Documents/code/tclustr/data/filtered/230927_nettcr_positives_only.csv -pad -20 -enc BL50LO -ne 20000 -cuda False -lwseq 3 -lwkld 1e-2 -lwtrp 0.8 -dist_type cosine -margin 0.125 -mla1 7 -mla2 8 -mla3 22 -mlb1 6 -mlb2 7 -mlb3 23 -nl 64 -nh 128 -bs 512 -lr 5e-4 -wd 1e-4 -wu 15 -o 240110_FullTCR_PepWeightedCosTrp -rid FCoPw -kf 0 -seed 0 -pepweight True
