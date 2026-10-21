"""Small source-selected classifier comparison for the non-pretrained features."""
import json,sys
from pathlib import Path
import numpy as np,joblib
from sklearn.ensemble import HistGradientBoostingClassifier,ExtraTreesClassifier
from sklearn.utils import compute_sample_weight
from run_profiles import ROOT,PROJECT,region,choose,eval_one,save
from profile_features import profile_features
from barnacles.modern import prepare_cases
from barnacles.core import propose,reference_targets
cases=prepare_cases(PROJECT/'data');X={}
for k,c in cases.items():
 old=ROOT/f'cache/profile{k}.npz'
 if old.exists():
  z=np.load(old);c['p']=z['p'];X[k]=z['profile_hog']
 else:
  c['p'],_=propose(c['im']);_,X[k]=profile_features(c['im'],c['p'])
locked=[];full=[]
for source in [1,2]:
 c=cases[source];tr=region(c['p'],(0,0,368,1024));va=region(c['p'],(656,0,1024,1024));y=reference_targets(c['p'][tr],c['gt'][region(c['gt'],(0,0,368,1024))]);keep=y>=0;weights=compute_sample_weight('balanced',y[keep])
 models={'hgb9':joblib.load(ROOT/f'models/from{source}_profile_hog.joblib')['classifier'],
 'hgb31':HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=31,learning_rate=.05,min_samples_leaf=15,l2_regularization=2.,early_stopping=False,random_state=42),
 'extra_trees':ExtraTreesClassifier(n_estimators=160,max_depth=16,min_samples_leaf=3,max_features=.2,class_weight='balanced',random_state=42,n_jobs=1)}
 choices=[]
 for name,model in models.items():
  if name!='hgb9':model.fit(X[source][tr][keep],y[keep],sample_weight=weights if name=='hgb31' else None)
  v=choose(c['p'][va],model.predict_proba(X[source][va])[:,1],c['gt'][region(c['gt'],(656,0,1024,1024))]);choices.append((v['f1'],v['precision'],name,v,model));joblib.dump(dict(classifier=model,threshold=v['threshold'],features='profile_hog'),ROOT/f'models/tuned_from{source}_{name}.joblib',compress=3)
 winner=max(choices,key=lambda x:x[:2]);locked.append(dict(source=source,chosen=winner[2],validation=winner[3]));print('LOCKED',source,winner[2],winner[3]['f1'],flush=True)
 for _,_,name,v,model in choices:
  stats,_=eval_one(model,X[3-source],cases[3-source],(0,0,1024,1024),v['threshold']);full.append(dict(source=source,target=3-source,classifier=name,selected=name==winner[2],validation_f1=v['f1'],**stats));print('TRANSFER',source,name,stats['f1'],flush=True)
save(ROOT/'results/profile_classifier_selection.json',locked);save(ROOT/'results/profile_classifier_comparison.json',full)
