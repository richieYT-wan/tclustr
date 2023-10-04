#!/bin/bash

# 2309
source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate pynn

HOMEDIR=/home/projects/vaccine/people/yatwan/tclustr/
PYDIR=${HOMEDIR}pyscripts/
filename=${filename}
cd ${PYDIR}
python3 ./paired_vae_cdr3_vj_pep.py -f /home/projects/vaccine/people/yatwan/tclustr/data/filtered/230927_nettcr_positives_only.csv -pad -20 -enc BL50LO -mlb 25 -mla 24 -mlp 12 -ne 1000 -lwb 2 -lwa 2 -lwp 1 -lwkld 1.5 -cdr3b "TRB_CDR3" -cdr3a "TRA_CDR3" -pep "peptide" -v "None" -j "None" -nl 100 -nh 200 -lr 7.5e-4 -wd 1e-4 -o SingleTestPairedVAE -rid DELETE_ME -kf 1 -seed 1