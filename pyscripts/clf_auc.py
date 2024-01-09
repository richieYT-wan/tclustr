import glob
from sklearn.metrics import roc_auc_score
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm 

def get_auc_name(directory):
	preds = glob.glob(directory+'*/*valid_pred*.csv')
	preds = pd.concat([pd.read_csv(x) for x in preds])
	return {'auc':roc_auc_score(preds['binder'].values, preds['pred_prob'].values)}, directory.replace('TripletTuning_','').split('/')[-2]

maindir = '../output/clf_triplet_tuning/'
dirs = glob.glob(maindir+'*/')
output = Parallel(n_jobs=8)(delayed(get_auc_name)(directory=d) for d in tqdm(dirs))

df = pd.DataFrame(data=[x[0] for x in output], index=[x[1] for x in output])
df.to_csv('../output/clf_triplet_tuning/aucs.csv')
print(df.sort_values('auc',ascending=False).head(25))


