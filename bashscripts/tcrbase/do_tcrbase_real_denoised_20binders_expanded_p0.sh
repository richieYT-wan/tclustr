#! /usr/bin/bash

TCRBASE=/home/projects/vaccine/people/morni/tbcr_align/tbcr_align
DIRECTORY=/home/projects/vaccine/people/yatwan/tclustr/data/TCRBASE/tcrbase_expanded_denoised_p0_20binders_swapped/
OUTDIR=/home/projects/vaccine/people/yatwan/tclustr/output/TCRBASE/tcrbase_expanded_denoised_p0_20binders_swapped/

mkdir -p ${OUTDIR}
for f in $(ls ${DIRECTORY} | awk -F[_] '{print $1}' | sort -u)
do
  echo ${f}
  
  ${TCRBASE} -xs -w 1,1,4,1,1,4 -db "${DIRECTORY}${f}_db_p0.tsv" "${DIRECTORY}${f}_query_p0.tsv" > "${OUTDIR}${f}.txt"
  
done