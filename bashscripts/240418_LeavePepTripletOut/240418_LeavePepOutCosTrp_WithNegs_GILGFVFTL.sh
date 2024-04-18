source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate cuda

pep=GILGFVFTL
HOMEDIR=/home/projects/vaccine/people/yatwan/tclustr/
PYDIR=${HOMEDIR}pyscripts/
cd ${PYDIR}
python3 ./231102_fulltcr_tripletloss.py -f /home/projects/vaccine/people/yatwan/tclustr/data/filtered/240418_nettcr_expanded_20binders_17pep_with_immrepnegs.csv -pad -20 -enc BL50LO -ne 20000 -cuda True -lwseq 1 -lwkld 1e-2 -lwtrp 5 -dist_type cosine -margin 0.075 -mla1 7 -mla2 8 -mla3 22 -mlb1 6 -mlb2 7 -mlb3 23 -mlpep 0 -nl 100 -nh 256 -bs 1024 -lr 5e-4 -wd 1e-5 -wu 100 -o "WithNegs_leave_${pep}_out" -rid ${pep} -kf 0 -seed 0 -addpe True -bn True -ale True -ald True -ob False -pepweight False -posweight True -leave_pep_out ${pep} > "${HOMEDIR}logs/240418_WithNegs_leave_${pep}_out.log" 2>&1
