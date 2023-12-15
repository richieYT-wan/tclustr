#! /usr/bin/bash

# Run loops for triplets
nlatents=(32 64)
wklds=(1 0.1 0.01)

# ! to get idx
for nl in "${nlatents[@]}"; do
  if [ "$nl" -eq 32 ]; then
      walltime="05:15:00"
  elif [ "$nl" -eq 64 ]; then
      walltime="06:45:00"
  fi
  for wkld in "${wklds[@]}"; do
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
    outname="TripletTuning_NoTriplet_${nl}_0_${wkld}"
    for f in $(seq 0 4); do
      filename="${outname}_fold_${f}_${random_string}"
      script_content=$(cat <<EOF
source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate cuda

HOMEDIR=/home/projects/vaccine/people/yatwan/tclustr/
PYDIR=\${HOMEDIR}pyscripts/
filename=${filename}
cd \${PYDIR}
python3 ./231102_fulltcr_tripletloss.py -f /home/projects/vaccine/people/yatwan/tclustr/data/filtered/230927_nettcr_positives_only.csv -pad -20 -enc BL50LO -ne 27500 -cuda True -lwseq 3 -lwkld ${wkld} -lwtrp 0 -dist_type 'cosine' -margin None -mla1 0 -mla2 0 -mlb1 0 -mlb2 0 -nl ${nl} -nh $((nl * 2)) -bs 512 -lr 1.75e-4 -wd 1e-4 -wu 10 -o ${outname} -rid ${random_string} -kf ${f} -seed ${f}
EOF
)
        # Write the script content to a file
        echo "$script_content" > "/home/projects/vaccine/people/yatwan/tclustr/bashscripts/231201_triplet_tuning/${filename}.sh"
        chmod +x "/home/projects/vaccine/people/yatwan/tclustr/bashscripts/231201_triplet_tuning/${filename}.sh"
        qsub -W group_list=vaccine -A vaccine -m e -l nodes=1:gpus=1:ppn=40,mem=120gb,walltime=${walltime} "/home/projects/vaccine/people/yatwan/tclustr/bashscripts/231201_triplet_tuning/${filename}.sh"

    done
    movescript=$(cat <<EOF
cd /home/projects/vaccine/people/yatwan/tclustr/output/
ODIR=${outname}_${random_string}/
mkdir -p \${ODIR}
mv *${random_string}* \${ODIR}
EOF
)
    echo "$movescript" > "/home/projects/vaccine/people/yatwan/tclustr/bashscripts/move_${random_string}.sh"
  done
done