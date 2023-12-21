#!/bin/bash

# Define the characters that can be used
characters="abcdefghijkmnopqrstuvwxyzABCDEFGHJKLMNOPQRSTUVWXYZ0123456789"
# Generate a random index between 0 and 61 (total number of characters)
index=$((RANDOM % 60))
# Get the character at the generated index
first_char="${characters:index:1}"
# Generate the remaining 4 characters as a combination of the defined characters
rest_chars=$(head /dev/urandom | tr -dc "$characters" | head -c 4)
# Combine the first and remaining characters
random_string="${first_char}${rest_chars}"
outname="FullTCR_BimodalVAECLF_NoWarmUp_1250e"


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
python3 ./231208_bimodal_vae.py -f /home/projects/vaccine/people/yatwan/tclustr/data/filtered/231205_nettcr_old_26pep_with_swaps.csv -pad -20 -enc BL50LO -cuda True -ne 5000 -lwseq 3 -lwkld 1e-2 -lwvae 1 -lwtrp .8 -lwclf 1 -dist_type cosine -margin 0.125 -mla1 7 -mla2 8 -mlb1 6 -mlb2 7 -nl 64 -nh 128 -nhclf 64 -do 0.2 -bn True -n_layers 1 -bs 1024 -lr 1.25e-4 -wd 1e-4 -wu 15 -pepenc BL50LO -o ${outname} -rid ${random_string} -kf ${f} -seed ${f}
EOF
)
                              # Write the script content to a file
                              echo "$script_content" > "/home/projects/vaccine/people/yatwan/tclustr/bashscripts/${filename}.sh"
                              chmod +x "/home/projects/vaccine/people/yatwan/tclustr/bashscripts/${filename}.sh"
                              qsub -W group_list=vaccine -A vaccine -m e -l nodes=1:gpus=1:ppn=20,mem=120gb,walltime=${1} "/home/projects/vaccine/people/yatwan/tclustr/bashscripts/${filename}.sh"
#                              rm "/home/projects/vaccine/people/yatwan/tclustr/bashscripts/${filename}.sh"

done


movescript=$(cat <<EOF
cd /home/projects/vaccine/people/yatwan/tclustr/output/
ODIR=${outname}_${random_string}/
mkdir -p \${ODIR}
mv *${random_string}* \${ODIR}
EOF
)

echo "$movescript" > "/home/projects/vaccine/people/yatwan/tclustr/bashscripts/move_${random_string}.sh"


