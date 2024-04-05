source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate cuda

HOMEDIR=/home/projects/vaccine/people/yatwan/tclustr/
PYDIR=${HOMEDIR}pyscripts/

filename="BSSVAE_CDR3only_LongIsh_nhtcr256_nhpep128_nl100"
cd ${PYDIR}
python3 240313_BSSVAE.py -f ../data/multimodal/240326_nettcr_paired_NOswaps.csv -cuda True -pad -20 -enc BL50LO -addpe False -nhtcr 256 -nhpep 128 -nl 100 -act selu -do 0. -lr 3e-4 -bs 1024 -tol 1e-3 -lwseq 2 -lwkld_z 1 -mla1 0 -mla2 0 -mlb1 0 -mlb2 0 -debug False -pepweight False -kf 0 -seed 0 -o ${filename} -ne 15000 -wukld 1000 -kldts 0.01 -fp 100 -kld_dec 5e-4 -wd 1e-6 -addkldn False -lwkld_n 1e-2 -device cuda:0 -pair_only True -bn True -al_e True -al_d True > "${HOMEDIR}logs/$(date '+%y%m%d_%H%M')_${filename}.log" 2>&1
