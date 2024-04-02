source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate cuda

HOMEDIR=/home/projects/vaccine/people/yatwan/tclustr/
PYDIR=${HOMEDIR}pyscripts/

add_layer_enc=True
add_layer_dec=True
batchnorm=True
filename="BSSVAE_UNPAIRED_allTrue_LONG_wd5e-5_lwseq2_nhtcr256_nhpep128_nl100"
cd ${PYDIR}
python3 240313_BSSVAE.py -f ../data/multimodal/240314_multimodal_NO_HUMAN_tcr_pep.csv -cuda True -pad -20 -enc BL50LO -addpe False -nhtcr 256 -nhpep 128 -nl 100 -act selu -do 0.25 -lr 1e-4 -bs 1024 -tol 1e-3 -lwseq 2 -lwkld_z 1  -debug False -pepweight False -kf 0 -seed 0 -o ${filename} -ne 20000 -wukld 800 -kldts 0.01 -fp 100 -kld_dec 5e-4 -wd 5e-5 -addkldn False -lwkld_n 1e-2 -device cuda:0 -pair_only False -bn ${batchnorm} -al_e ${add_layer_enc} -al_d ${add_layer_dec} > "${HOMEDIR}logs/$(date '+%y%m%d_%H%M')_${filename}.log" 2>&1
