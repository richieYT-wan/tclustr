#! /usr/bin/bash

TCRBASE=/home/projects/vaccine/people/morni/tbcr_align/tbcr_align
DIRECTORY=/home/projects/vaccine/people/yatwan/tclustr/data/tcrbase_expanded_p0_swapped/
OUTDIR=/home/projects/vaccine/people/yatwan/tclustr/output/tcrbase_expanded_p0_swapped/

mkdir -p ${OUTDIR}
for f in $(ls ${DIRECTORY} | awk -F[_] '{print $1}' | sort -u)
do
  echo ${f}
  
  ${TCRBASE} -xs -w 0,0,4,0,0,4 -db "${DIRECTORY}${f}_db_p0.tsv" "${DIRECTORY}${f}_query_p0.tsv" > "${OUTDIR}${f}.txt"
done