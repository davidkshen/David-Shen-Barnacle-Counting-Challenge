import sys,json,time,hashlib
from pathlib import Path
import numpy as np,joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils import compute_sample_weight
sys.path.insert(0,str(Path(__file__).resolve().parent))
from profile_features import profile_features
from barnacles.modern import prepare_cases
from barnacles.core import propose,reference_targets,select_detections,metrics,suppress
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT/'project'
if not PROJECT.exists():PROJECT=ROOT.parent/'recovered/project/barnacle_assistant'
for name in ['results','models','cache']:(ROOT/name).mkdir(exist_ok=True)

def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def region(p,b):return (p[:,0]>=b[0])&(p[:,0]<b[2])&(p[:,1]>=b[1])&(p[:,1]<b[3])
def fit(X,y,local=None):
    keep=y>=0;yy=y[keep];w=compute_sample_weight('balanced',yy)
    if local is not None:
        ll=local[keep]
        if ll.any():
            # Equal overall mass to original-source and new-image examples;
            # each domain separately balances its positive/negative classes.
            for mask in [ll,~ll]:w[mask]=compute_sample_weight('balanced',yy[mask])/mask.sum()
    clf=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=9,learning_rate=.07,min_samples_leaf=25,l2_regularization=2.,early_stopping=False,random_state=42)
    return clf.fit(X[keep],yy,sample_weight=w*(len(w)/w.sum()))
def choose(p,s,gt):
    ids=suppress(p[:,:2],s,distance=4.,radii=p[:,2]);v=s[ids]
    ts=np.unique(np.r_[0.,1.000001,np.quantile(v,np.linspace(.1,.995,30))])
    rows=[dict(threshold=float(t),**metrics(p[ids[v>=t]],gt)) for t in ts]
    return max(rows,key=lambda x:(x['f1'],x['precision']))
def eval_one(clf,X,c,b,t):
    keep=region(c['p'],b);pp=c['p'][keep];s=clf.predict_proba(X[keep])[:,1];ids=select_detections(pp,s,t)
    gt=c['gt'][region(c['gt'],b)];pred=pp[ids]
    return dict(**metrics(pred,gt),strict_center_f1=metrics(pred,gt,.5)['f1']),pred

def main():
    cases=prepare_cases(PROJECT/'data');f={}
    for k,c in cases.items():
        version=hashlib.sha256((ROOT/'profile_features.py').read_bytes()+c['im'].tobytes()).hexdigest()[:12]
        cache=ROOT/f'cache/profile{k}_{version}.npz'
        if cache.exists():
            z=np.load(cache);c['p']=z['p'];f[k]={'profile':z['profile'],'profile_hog':z['profile_hog']}
        else:
            c['p'],_=propose(c['im']);a,b=profile_features(c['im'],c['p']);f[k]={'profile':a,'profile_hog':b}
            np.savez_compressed(cache,p=c['p'],profile=a,profile_hog=b)
        print('FEATURES',k,{m:x.shape for m,x in f[k].items()},flush=True)
    rows=[];models={}
    for source in (1,2):
        c=cases[source];tr=region(c['p'],(0,0,368,1024));va=region(c['p'],(656,0,1024,1024))
        y=reference_targets(c['p'][tr],c['gt'][region(c['gt'],(0,0,368,1024))]);vg=c['gt'][region(c['gt'],(656,0,1024,1024))]
        choices=[]
        for name,X in f[source].items():
            clf=fit(X[tr],y);v=choose(c['p'][va],clf.predict_proba(X[va])[:,1],vg)
            joblib.dump(dict(classifier=clf,threshold=v['threshold'],features=name),ROOT/f'models/from{source}_{name}.joblib',compress=3)
            stats,pred=eval_one(clf,f[3-source][name],cases[3-source],(0,0,1024,1024),v['threshold'])
            row=dict(source=source,target=3-source,method=name,validation=v,**stats);rows.append(row)
            choices.append((v['f1'],v['precision'],name,clf,v));print('TRANSFER',row,flush=True)
        _,_,name,clf,v=max(choices,key=lambda x:x[:2]);models[source]=(name,clf,v)
    save(ROOT/'results/profile_transfer.json',rows)
    hybrid=[];manifest=[]
    test=(832,0,1024,1024);cal=(512,384,640,512)
    for source in (1,2):
        name,base,locked=models[source];target=3-source;c=cases[source];t=cases[target];Xs=f[source][name];Xt=f[target][name]
        src=region(c['p'],(0,0,368,1024));sy=reference_targets(c['p'][src],c['gt'][region(c['gt'],(0,0,368,1024))])
        base_stats,_=eval_one(base,Xt,t,test,locked['threshold'])
        cg=t['gt'][region(t['gt'],cal)];ci=region(t['p'],cal)
        # Calibrate the original model on the same small patch as a lower-cost
        # alternative to retraining. It uses no labels from the reserved stripe.
        calibrated=choose(t['p'][ci],base.predict_proba(Xt[ci])[:,1],cg)
        cs,_=eval_one(base,Xt,t,test,calibrated['threshold'])
        hybrid.append(dict(source=source,target=target,method='threshold_only',budget_tiles=0,annotation_labels=len(cg),baseline=base_stats,**cs))
        tiles=[(x,y,x+128,y+128) for x in (0,128) for y in (0,256,512,768)]
        for seed in (42,43,44):
            order=np.random.default_rng(seed).permutation(len(tiles))
            for budget in (1,2,4):
                picked=[tiles[j] for j in order[:budget]];ii=np.zeros(len(t['p']),bool);yy=np.full(len(t['p']),-1,np.int8);nlabels=0
                for b in picked:
                    mask=region(t['p'],b);gg=t['gt'][region(t['gt'],b)];nlabels+=len(gg)
                    ii|=mask;yy[mask]=reference_targets(t['p'][mask],gg) if len(gg) else 0
                X=np.concatenate([Xs[src],Xt[ii]]);y=np.r_[sy,yy[ii]];local=np.r_[np.zeros(src.sum(),bool),np.ones(ii.sum(),bool)]
                if len(np.unique(y[local&(y>=0)]))<2:
                    hybrid.append(dict(source=source,target=target,method='adapt',seed=seed,budget_tiles=budget,status='local patch lacks both classes'));continue
                model=fit(X,y,local);selected=choose(t['p'][ci],model.predict_proba(Xt[ci])[:,1],cg)
                stats,pred=eval_one(model,Xt,t,test,selected['threshold'])
                row=dict(source=source,target=target,method='adapt',seed=seed,budget_tiles=budget,training_labels=nlabels,calibration_labels=len(cg),annotation_labels=nlabels+len(cg),calibration=selected,baseline=base_stats,**stats)
                hybrid.append(row);manifest.append(dict(source=source,target=target,seed=seed,budget=budget,train=picked,calibration=cal,test=test))
                joblib.dump(dict(classifier=model,threshold=selected['threshold'],features=name,protocol=manifest[-1]),ROOT/f'models/adapt{target}_{seed}_{budget}.joblib',compress=3)
                print('ADAPT',source,target,seed,budget,'labels',nlabels+len(cg),'F1',stats['f1'],flush=True)
    save(ROOT/'results/profile_hybrid.json',hybrid);save(ROOT/'results/profile_hybrid_manifest.json',manifest)

if __name__=='__main__':main()
