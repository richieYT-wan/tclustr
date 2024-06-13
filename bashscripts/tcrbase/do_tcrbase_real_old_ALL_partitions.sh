#! /usr/bin/bash

TCRBASE=/home/projects/vaccine/people/morni/tbcr_align/tbcr_align
DIRECTORY=/home/projects/vaccine/people/yatwan/tclustr/data/TCRBASE/tcrbase_old_ALL_swapped/
OUTDIR=/home/projects/vaccine/people/yatwan/tclustr/output/TCRBASE/tcrbase_old_ALL_swapped/

mkdir -p ${OUTDIR}
for f in $(ls ${DIRECTORY} | awk -F[_] '{print $1}' | sort -u)
do
  echo ${f}
  for i in 0 1 2 3 4
  do
    ${TCRBASE} -xs -w 1,1,4,1,1,4 -db "${DIRECTORY}${f}_db_p${i}.tsv" "${DIRECTORY}${f}_query_p${i}.tsv" > "${OUTDIR}${f}_p${i}.txt"
  done
done