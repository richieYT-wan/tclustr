#!/bin/bash

run_id="COS2X"
outname="231205_TripletCosine_A3B3_expanded_data"

for f in $(seq 0 4);
do
 filename="${outname}_fold_${f}_${run_id}"
  script_content=$(cat <<EOF
source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate pynn

HOMEDIR=/home/projects/vaccine/people/yatwan/tclustr/
PYDIR=\${HOMEDIR}pyscripts/
filename=${filename}
cd \${PYDIR}
python3 ./231102_fulltcr_tripletloss.py -f /home/projects/vaccine/people/yatwan/tclustr/data/filtered/231205_nettcr22_expanded_a3b3_pos_only.csv -pad -20 -enc BL50LO -ne 27500 -lwseq 3 -lwkld 5e-3 -lwtrp 0.75 -dist_type cosine -margin 0.1 -mla1 0 -mla2 0 -mlb1 0 -mlb2 0 -nl 64 -nh 128 -bs 512 -lr 1e-4 -wd 1e-4 -wu 15 -o ${outname} -rid ${run_id} -kf ${f} -seed ${f}
EOF
)
                              # Write the script content to a file
                              echo "$script_content" > "/home/projects/vaccine/people/yatwan/tclustr/bashscripts/${filename}.sh"
                              chmod +x "/home/projects/vaccine/people/yatwan/tclustr/bashscripts/${filename}.sh"
                              qsub -W group_list=vaccine -A vaccine -m e -l nodes=1:ppn=5:thinnode,mem=36gb,walltime=${1} "/home/projects/vaccine/people/yatwan/tclustr/bashscripts/${filename}.sh"
                              rm "/home/projects/vaccine/people/yatwan/tclustr/bashscripts/${filename}.sh"

done


movescript=$(cat <<EOF
cd /home/projects/vaccine/people/yatwan/tclustr/output/
ODIR=${outname}_${run_id}/
mkdir -p \${ODIR}
mv *${run_id}* \${ODIR}
EOF
)

echo "$movescript" > "/home/projects/vaccine/people/yatwan/tclustr/bashscripts/move_${run_id}.sh"


