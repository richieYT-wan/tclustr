#! /usr/bin/bash
for seed in $(seq -f "%03g" 0 99);
do

  filename="IMMREP25_Ratio1_Run${seed}"
script_content=$(cat <<EOF
#! /usr/bin/bash
source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate cuda
HOMEDIR=/home/projects/vaccine/people/yatwan/tclustr/
PYDIR=\${HOMEDIR}pyscripts/
filename=${filename}
cd \${PYDIR}
python3 250306_cluster_against_healthy.py -seed ${seed} -o ${filename} -rid IMMREP25 -ratio 1 -n_jobs 4
EOF
)
echo "$script_content" > "./250306_IMMREP25_v_garner/${filename}.sh"
chmod +x "./250306_IMMREP25_v_garner/${filename}.sh"
done

