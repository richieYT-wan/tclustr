#! /usr/bin/bash

TCRBASE=/home/projects/vaccine/people/morni/tbcr_align/tbcr_align
DIRECTORY=/home/projects/vaccine/people/yatwan/tclustr/data/tcrbase_top10pep/
OUTDIR=/home/projects/vaccine/people/yatwan/tclustr/output/tcrbase_top10pep/

mkdir -p ${OUTDIR}
for pep in GILGFVFTL RAKFKQLL KLGGALQAK AVFDRKSDAK ELAGIGILTV NLVPMVATV IVTDFSVIK LLWNGPMAV CINGVCWTV GLCTLVAML;
do
  for fold in $(seq 0 4);
  do
    ${TCRBASE} -xs -w 0,0,4,0,0,4 -db "${DIRECTORY}${pep}_train_fold_${fold}.tsv" "${DIRECTORY}${pep}_valid_fold_${fold}.tsv" > "${OUTDIR}${pep}_fold_${fold}"
  done
done