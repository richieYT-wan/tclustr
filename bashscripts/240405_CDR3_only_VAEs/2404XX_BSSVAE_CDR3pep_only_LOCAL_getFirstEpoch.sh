
HOMEDIR=/Users/riwa/Documents/code/tclustr/
PYDIR=${HOMEDIR}pyscripts/

filename="BSSVAE_CDR3ONLY_LOCAL"
cd ${PYDIR}
python3 240313_BSSVAE.py -f ${HOMEDIR}data/multimodal/240326_nettcr_paired_NOswaps.csv -cuda True -pad -20 -enc BL50LO -addpe False -nhtcr 256 -nhpep 128 -nl 100 -act selu -do 0. -lr 3e-4 -bs 1024 -tol 1e-3 -lwseq 2 -lwkld_z 1 -mla1 0 -mla2 0 -mlb1 0 -mlb2 0 -debug False -pepweight False -kf 0 -seed 0 -o ${filename} -ne 5 -wukld 1000 -kldts 0.01 -fp 100 -kld_dec 5e-4 -wd 1e-6 -addkldn False -lwkld_n 1e-2 -device cpu -pair_only True -bn True -al_e True -al_d True > "${HOMEDIR}logs/XX.log" 2>&1
