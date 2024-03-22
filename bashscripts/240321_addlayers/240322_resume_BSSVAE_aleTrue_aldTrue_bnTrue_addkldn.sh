source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate cuda

HOMEDIR=/home/projects/vaccine/people/yatwan/tclustr/
PYDIR=${HOMEDIR}pyscripts/

add_layer_enc=True
add_layer_dec=True
batchnorm=True
filename="RESUME_ADDKLDN_BSSVAE_addlenc${add_layer_enc}_addldec${add_layer_dec}_bn${batchnorm}_2p5kepochs_wd5e-5_lwseq2_nhtcr256_nhpep128_nl100"
cd ${PYDIR}
python3 240321_resume_multimodal.py -model_folder ../output/240321_1427_BSSVAE_addlencTrue_addldecTrue_bnTrue_2p5kepochs_wd5e-5_lwseq2_nhtcr256_nhpep128_nl100_KFold_0_VoZuWy/ -o ${filename} -wd 1e-8 -ne 10000 -cuda True -device cuda:0 "${HOMEDIR}logs/$(date '+%y%m%d_%H%M')_${filename}.log" 2>&1
