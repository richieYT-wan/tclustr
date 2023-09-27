#!/bin/bash

# 230927_0942 : Redo the runs with the wrong v/j dim now that it's fixed
# USING THE PREVIOUS SEQ/KLD/V/J WEIGHTS: 3, 1, 2.5, 2
# Define the characters that can be used

random_string='bb76k'
outname="FirstTestKFOLD_${random_string}"


for f in 0 3;
do
 filename="${outname}_fold_${f}_${random_string}"
  script_content=$(cat <<EOF
source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate pynn

HOMEDIR=/home/projects/vaccine/people/yatwan/tclustr/
PYDIR=\${HOMEDIR}pyscripts/
filename=${filename}
cd \${PYDIR}
python3 ./vae_cdr3_vj.py -f /home/projects/vaccine/people/yatwan/tclustr/data/filtered/230921_nettcr_immrepnegs_noswap.csv -pad -20 -enc BL50LO -ml 23 -ne 1000 -lwseq 3 -lwkld 1 -lwv 2.5 -lwj 2 -o ${outname} -rid ${random_string} -kf ${f} -seed ${f}
EOF
)
                              # Write the script content to a file
                              echo "$script_content" > "/home/projects/vaccine/people/yatwan/tclustr/bashscripts/${filename}.sh"
                              chmod +x "/home/projects/vaccine/people/yatwan/tclustr/bashscripts/${filename}.sh"
                              qsub -W group_list=vaccine -A vaccine -m e -l nodes=1:ppn=5:thinnode,mem=36gb,walltime=${1} "/home/projects/vaccine/people/yatwan/tclustr/bashscripts/${filename}.sh"
                              rm "/home/projects/vaccine/people/yatwan/tclustr/bashscripts/${filename}.sh"

done
