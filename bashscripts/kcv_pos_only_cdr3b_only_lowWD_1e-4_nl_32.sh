#!/bin/bash

# 230927_0942 : REDO THE RUNS!
# What changed : default V and J map, readjusted seq/kld/v/j/ losses weights (3, 2, 2.5, 1.5)
# Lowered N_epochs because 2000 was too much (do 1750)

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
outname="OnlyPositivesFullCDR3b_LowerDim_32_WD_1e-4"


for f in $(seq 0 4);
do
 filename="${outname}_fold_${f}_${random_string}"
  script_content=$(cat <<EOF
source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate pynn

HOMEDIR=/home/projects/vaccine/people/yatwan/tclustr/
PYDIR=\${HOMEDIR}pyscripts/
filename=${filename}
cd \${PYDIR}
python3 ./vae_cdr3_vj.py -f /home/projects/vaccine/people/yatwan/tclustr/data/filtered/230927_nettcr_positives_only.csv -pad -20 -enc BL50LO -ml 25 -ne 2000 -lwseq 2 -lwkld 1 -cdr3 "TRB_CDR3" -v "None" -j "None" -nl 32 -nh 64 -lr 5e-4 -wd 1e-4 -o ${outname} -rid ${random_string} -kf ${f} -seed ${f}
EOF
)
                              # Write the script content to a file
                              echo "$script_content" > "/home/projects/vaccine/people/yatwan/tclustr/bashscripts/${filename}.sh"
                              chmod +x "/home/projects/vaccine/people/yatwan/tclustr/bashscripts/${filename}.sh"
                              qsub -W group_list=vaccine -A vaccine -m e -l nodes=1:ppn=5:thinnode,mem=36gb,walltime=${1} "/home/projects/vaccine/people/yatwan/tclustr/bashscripts/${filename}.sh"
                              rm "/home/projects/vaccine/people/yatwan/tclustr/bashscripts/${filename}.sh"

done
