source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate pynn

HOMEDIR=/home/projects/vaccine/people/yatwan/tclustr/
PYDIR=\${HOMEDIR}pyscripts/
filename=${filename}
cd \${PYDIR}
python3 ./vae_cdr3_vj.py -f /home/projects/vaccine/people/yatwan/tclustr/data/filtered/230921_nettcr_immrepnegs_noswap.csv -pad -20 -enc BL50LO -ml 23 -ne 1750 -lwseq 3 -lwkld 2 -lwv 2.5 -lwj 1.5 -o "RedoTestKFOLD" -rid 'rCSqS' -kf 4 -seed 4