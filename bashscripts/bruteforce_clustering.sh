source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate pynn
HOMEDIR=/home/projects/vaccine/people/yatwan/tclustr/
PYDIR=${HOMEDIR}pyscripts/
cd ${PYDIR}

echo "KMeans"
python3 bruteforce_clustering_KM.py
echo "DBScan"
python3 bruteforce_clustering_DBSCAN.py
echo "SC"
python3 bruteforce_clustering_SC.py