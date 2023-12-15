#!/bin/bash

# ARGS
# 1 : walltime


# Define the characters that can be used
characters="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
# Generate a random index between 0 and 61 (total number of characters)
index=$((RANDOM % 62))
# Get the character at the generated index
first_char="${characters:index:1}"
# Generate the remaining 4 characters as a combination of the defined characters
rest_chars=$(head /dev/urandom | tr -dc "$characters" | head -c 4)
# Combine the first and remaining characters
random_string="${first_char}${rest_chars}"
outname="CDR3_AB_nh_128_wd_1e-3_weights_3to1"


for f in $(seq 0 4);
do
 filename="${outname}_fold_${f}_${random_string}"
  script_content=$(cat <<EOF
source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate cuda

HOMEDIR=/home/projects/vaccine/people/yatwan/tclustr/
PYDIR=\${HOMEDIR}pyscripts/
filename=${filename}
cd \${PYDIR}
python3 ./fulltcr_vae.py -f /home/projects/vaccine/people/yatwan/tclustr/data/filtered/230927_nettcr_positives_only.csv -pad -20 -mla1 0 -mla2 0 -mlb1 0 -mlb2 0 -enc BL50LO -ne 25000 -lwseq 3 -lwkld 1e-2 -nl 64 -nh 128 -lr 5e-4 -wd 1e-4 -wu 15 -o ${outname} -rid ${random_string} -kf ${f} -seed ${f}
EOF
)
                              # Write the script content to a file
                              echo "$script_content" > "/home/projects/vaccine/people/yatwan/tclustr/bashscripts/${filename}.sh"
                              chmod +x "/home/projects/vaccine/people/yatwan/tclustr/bashscripts/${filename}.sh"
                              qsub -W group_list=vaccine -A vaccine -m e -l nodes=1:gpus=1:ppn=40,mem=36gb,walltime=${1} "/home/projects/vaccine/people/yatwan/tclustr/bashscripts/${filename}.sh"
                              rm "/home/projects/vaccine/people/yatwan/tclustr/bashscripts/${filename}.sh"

done


movescript=$(cat <<EOF
cd /home/projects/vaccine/people/yatwan/tclustr/output/
ODIR=${outname}_${random_string}/
mkdir -p \${ODIR}
mv *${random_string}* \${ODIR}
EOF
)

echo "$movescript" > "/home/projects/vaccine/people/yatwan/tclustr/bashscripts/move_${random_string}.sh"


