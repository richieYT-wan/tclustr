#! /usr/bin/bash
source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate pynn

HOMEDIR=/home/projects/vaccine/people/yatwan/tclustr/
PYDIR=${HOMEDIR}pyscripts/
cd ${PYDIR}

for f in 'FullTCR_nh_128_wd_1e-3_weights_3to1_8NoCI' 'FullTCR_nh_128_wd_1e-4_weights_3to1_YTsp8' 'FullTCR_nh_256_wd_1e-4_weights_3to1_0UeYG'
do
  python3 redo_extract_agg_allfolds.py