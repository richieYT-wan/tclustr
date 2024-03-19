source /home/people/riwa/anaconda3/etc/profile.d/conda.sh
source activate cuda

HOMEDIR=/home/projects2/riwa/tclustr/
PYDIR=${HOMEDIR}pyscripts/
filename=BSSVAE_paironly_wd1e-3_5kepochs
cd ${PYDIR}
python3 240313_BSSVAE.py -f ../data/multimodal/240314_tcrpep_df.csv -cuda True -pad -20 -enc BL50LO -addpe False -nhtcr 200 -nhpep 150 -nl 100 -act selu -do 0.25 -bn True -lr 1e-4 -wd 1e-3 -bs 1024 -tol 1e-5 -lwseq 1 -lwkld_n 0.1 -lwkld_z 1 -ne 10000 -wukld 250 -kldts 0.08 -fp 50 -kld_dec 3e-3 -debug False -pepweight False -kf 0 -seed 0 -o ${filename} -pair_only True
