#! /usr/bin/bash

# HERE IT DEPENDS ON WHETHER WE ARE USING C2 OR HTC
source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate cuda

# Define the characters that can be used
characters="abcdefghijkmnopqrstuvwxyzABCDEFGHJKLMNOPQRSTUVWXYZ0123456789"
# Generate a random index between 0 and 61 (total number of characters)
index=$((RANDOM % 60))
# Get the character at the generated index
first_char="${characters:index:1}"
# Generate the remaining 4 characters as a combination of the defined characters
rest_chars=$(head /dev/urandom | tr -dc "$characters" | head -c 4)
# Combine the first and remaining characters
random_id="${first_char}${rest_chars}"

# Setting input directory paths
HOMEDIR=/home/projects/vaccine/people/yatwan/tclustr/
PYDIR=${HOMEDIR}pyscripts/

HOMEDIR='/home/projects/vaccine/people/yatwan/tclustr/'
PYDIR="${HOMEDIR}pyscripts/"
BASHDIR="${HOMEDIR}bashscripts/"
C2PATH="/home/projects/vaccine/people/morni/bin/tbcr_align"
HTCPATH="/home/people/morni/bin/tbcr_align"
CHAINS=("A1" "A2" "A3" "B1" "B2" "B3")  # Default chains
LABELCOL="peptide"
EXTRACOLS=("original_peptide" "binder" "partition" "original_index")
TBCRALIGN="$HTCPATH"
chainarg="full"
# args : 1 = mainfolder ; 2 = outfolder ; 3 = n_epochs ; 4 = grep
while getopts ":f:c:s:l:e" opt; do
  case ${opt} in
    f )
      INPUTFILE=$OPTARG
      ;;
    c )
      # If -c is used, override the default chains
      CHAINS=("$OPTARG")  # Add the first option after -c
      while [[ ${!OPTIND} =~ ^[^-] ]]; do
        CHAINS+=("${!OPTIND}")
        OPTIND=$((OPTIND + 1))
      done
      ;;
    s )
      if [ "$OPTARG" == "htc" ]; then
        TBCRALIGN="$HTCPATH"
      elif [ "$OPTARG" == "c2" ]; then
        TBCRALIGN="$C2PATH"
      fi
      ;;
    e )
      EXTRACOLS=("$OPTARG")  # Add the first option after -c
      while [[ ${!OPTIND} =~ ^[^-] ]]; do
        EXTRACOLS+=("${!OPTIND}")
        OPTIND=$((OPTIND + 1))
      done
      ;;
    l )
      LABELCOL="$OPTARG"  # Add the first option after -c
      ;;
    \? )
      echo "Usage: $0 -f <INPUTFILE> -c <CHAINS> (ex: A1 A2 A3 B1 B2 B3) -s <SERVER> (c2/htc) -l <LABELCOL> -e <EXTRACOLS>"
      exit 1
      ;;
    : )
      echo "Invalid option: -$OPTARG requires an argument"
      exit 1
      ;;
  esac
done


# TODO DEFINE A SINGLE OUTDIR WHERE WE SAVE ALL THE OUTPUT DMs
cd $BASHDIR
sh do_tbcralign.sh -f $INPUTFILE-c A1 A2 A3 B1 B2 B3 -s c2 -l Disease -e Disease Source count index_col Run

cd $PYDIR
python3 do_tcrdist.py -f $INPUTFILE -pep $LABELCOL -others Disease CancerType count -idx index_col

# TODO FINISH THIS
python3 240819_MST_cuts_clustering.py -... options


# Get tcrdist3 dist_matrix
# Get TBCRalign dist_matrix
# Run pipeline and read both baseline DMs, read df --> get model --> latent --> VAE dm --> run MST size vs topn vs AggloCluster for all 3 matrices
# --> Do silhouette plot (one for each dm) and total final RetPur curves

