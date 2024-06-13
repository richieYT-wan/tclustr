source /home/people/riwa/anaconda3/etc/profile.d/conda.sh
source activate cuda

HOMEDIR=/home/projects2/riwa/tclustr/
PYDIR=${HOMEDIR}pyscripts/
filename=JMVAE_10kepochs_wkld_1e-2_wd_5e-3_addkldn_True
cd ${PYDIR}
python3 240320_JMVAE.py -f ../data/multimodal/240314_multimodal_NO_HUMAN_tcr_pep.csv -cuda True -pad -20 -enc BL50LO -addpe False -nhtcr 200 -nhpep 150 -nl 100 -act selu -do 0.25 -bn True -lr 1e-4 -bs 1024 -tol 1e-3 -lwseq 1 -lwkld_z 1  -debug False -pepweight False -kf 0 -seed 0 -o ${filename} -pair_only True -return_pair True -ne 10000 -wukld 800 -kldts 0.01 -fp 100 -kld_dec 5e-4 -wd 5e-3 -addkldn True -lwkld_n 1e-2 -device cuda:0 > "${HOMEDIR}logs/$(date '+%y%m%d_%H%M')_${filename}.log" 2>&1
