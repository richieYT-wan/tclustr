source /home/people/riwa/anaconda3/etc/profile.d/conda.sh
source activate cuda

HOMEDIR=/home/projects2/riwa/tclustr/
PYDIR=${HOMEDIR}pyscripts/
filename=BSSVAE_256nh_128nl_20kepochs_lower_LR_leakyrelu_5e-3_KLDN_KLDzAnneal_500WU
cd ${PYDIR}
python3 240313_BSSVAE.py -f ../data/multimodal/240314_multimodal_NO_HUMAN_tcr_pep.csv -cuda True -pad -20 -enc BL50LO -addpe False -nhtcr 256 -nhpep 200 -nl 128 -act leakyrelu -do 0.25 -bn True -lr 1e-4 -wd 1e-4 -bs 1024 -tol 1e-5 -lwseq 1 -lwkld_n 5e-3 -lwkld_z 1 -ne 20000 -wukld 500 -kldts 0.05 -fp 75 -kld_dec 5e-3 -debug False -pepweight False -kf 0 -seed 0 -o ${filename}
