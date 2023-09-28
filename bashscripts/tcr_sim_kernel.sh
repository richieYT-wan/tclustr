#! /usr/bin/bash

pepfile=${1}
pepdb=${2}
/home/projects/vaccine/people/morni/bin/pep2score_db_kernel -pa -t 2 ${pepfile} ${pepdb}