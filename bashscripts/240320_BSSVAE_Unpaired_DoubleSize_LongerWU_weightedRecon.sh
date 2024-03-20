source /home/people/riwa/anaconda3/etc/profile.d/conda.sh
source activate cuda

HOMEDIR=/home/projects2/riwa/tclustr/
PYDIR=${HOMEDIR}pyscripts/
filename=BSSVAE_wTCRrec_DoubleSize_longWU_Unpaired_wN1e-2_jointKLDn_leakyrelu
cd ${PYDIR}
python3 240313_BSSVAE.py -f ../data/multimodal/240314_multimodal_NO_HUMAN_tcr_pep.csv -cuda True -pad -20 -enc BL50LO -addpe False -nhtcr 512 -nhpep 256 -nl 256 -act leakyrelu -do 0.25 -bn True -lr 5e-4 -wd 5e-4 -bs 1024 -tol 1e-3 -lwseq 1.5 -lwkld_n 0.01 -lwkld_z 1.5 -ne 10000 -wukld 1000 -kldts 0.01 -fp 100 -kld_dec 1e-3 -debug False -pepweight False -kf 0 -seed 0 -o ${filename} -pair_only False -addkldn True -device cuda:0 > "${HOMEDIR}logs/$(date '+%y%m%d_%H%M')_${filename}.log" 2>&1
