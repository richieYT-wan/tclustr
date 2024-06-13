source /home/people/riwa/anaconda3/etc/profile.d/conda.sh
source activate cuda

HOMEDIR=/home/projects2/riwa/tclustr/
PYDIR=${HOMEDIR}pyscripts/


# filename=240110_Bimodal_FullTCR_PepWeightedCosClf_fold_1_BWuPw
# Using pepweights to set 0-1 on triplet loss
cd ${PYDIR}
python3 ./231208_bimodal_vae.py -f /home/projects2/riwa/tclustr/data/multimodal/240326_nettcr_paired_withswaps.csv -pad -20 -enc BL50LO -cuda True -ne 4000 -lwseq 1 -lwkld 1e-2 -lwvae 1 -lwtrp 0 -lwclf 1 -dist_type cosine -margin 0.075 -mla1 0 -mla2 0 -mlb1 0 -mlb2 0 -mlpep 0 -nl 64 -nh 128 -nhclf 64 -do 0.2 -n_layers 1 -bs 1024 -lr 5e-4 -wd 1e-5 -wu 100 -wuclf 2500 -pepenc BL50LO -o ExpData_CDR3ONLY_2stage_SMALL_128h_64l_NoTrp -rid 2stTCRPsmall -kf 0 -seed 0 -posweight True -pepweight True -addpe True -bn True -ale True -ald True -ob False -device cuda:1 > "${HOMEDIR}logs/240405_CDR3ONLY_2stage_small.log" 2>&1
