#! /usr/bin/bash
source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate pynn

HOMEDIR=/home/projects/vaccine/people/yatwan/tclustr/
PYDIR=${HOMEDIR}pyscripts/
DDIR=${HOMEDIR}output/2310XX_MoreChains/

cd ${PYDIR}

for f in 'CDR3_AB_nh_128_wd_1e-3_weights_3to1_i1RsR/' 'CDR3_AB_nh_128_wd_1e-4_weights_3to1_LlWGn/'
do
  python3 redo_extract_agg_allfolds_CDR3AB.py -d "${DDIR}${f}" -nh 128 -nl 64
done
