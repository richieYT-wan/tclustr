source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate cuda

pep=AllTriplet
HOMEDIR=/home/projects/vaccine/people/yatwan/tclustr/
PYDIR=${HOMEDIR}pyscripts/
cd ${PYDIR}
python3 ./231102_fulltcr_tripletloss.py -f /home/projects/vaccine/people/yatwan/tclustr/data/filtered/240418_nettcr_expanded_20binders_17pep_POSONLY.csv -pad -20 -enc BL50LO -ne 20000 -cuda True -lwseq 1 -lwkld 10 -lwtrp 3 -dist_type cosine -margin 0.2 -mla1 7 -mla2 8 -mla3 22 -mlb1 6 -mlb2 7 -mlb3 23 -mlpep 0 -nl 100 -nh 256 -bs 1024 -lr 5e-4 -wd 1e-5 -wu 150 -fp 50 -kld_dec 1e-2 -kldts 0.075 -o "RedoMargin_0200" -rid ${pep} -kf 0 -seed 0 -addpe True -bn True -ale True -ald True -ob False -pepweight False -posweight True > "${HOMEDIR}logs/240424_17Peps_RedoMargin_0200.log" 2>&1
