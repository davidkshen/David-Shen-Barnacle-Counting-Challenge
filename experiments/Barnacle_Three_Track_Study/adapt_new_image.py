"""Adapt a saved detector using fully annotated patches of one new photograph.

Boxes and patch bounds use the rectified 1024px quadrat coordinate system.
This consumes supervised labels; it does not infer live/dead biology or make
new-photo accuracy claims. It writes a separate model, preserving the base.
"""
import argparse,json,re
from pathlib import Path
import numpy as np
from run_detectors import ROOT,PROJECT,train,calibrate,save
from barnacles.core import read_rgb,rectify
from barnacles.modern import prepare_cases

def overlaps(a,b):return max(a[0],b[0])<min(a[2],b[2]) and max(a[1],b[1])<min(a[3],b[3])
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--image',required=True);ap.add_argument('--annotations',required=True);ap.add_argument('--base-model',required=True);ap.add_argument('--tag',required=True);ap.add_argument('--epochs',type=int,default=30);ap.add_argument('--replay-source',type=int,choices=[1,2]);ap.add_argument('--weight-origin',choices=['provided_images_only','external_pretraining','unknown'],default='unknown');a=ap.parse_args()
    if not re.fullmatch(r'[a-zA-Z0-9_-]+',a.tag):raise ValueError('Tag must contain only letters, numbers, underscore or hyphen')
    spec=json.loads(Path(a.annotations).read_text());rgb,_=rectify(read_rgb(a.image),spec['corners']);patches=spec['patches'];tr=[];va=[];boxes=[]
    for j,p in enumerate(patches):
        b=p['bounds']
        if len(b)!=4 or any(int(v)!=v for v in b) or not (0<=b[0]<b[2]<=1024 and 0<=b[1]<b[3]<=1024):raise ValueError('Patch bounds must be integer rectangles inside 1024px image')
        if p.get('complete') is not True:raise ValueError('Every patch must confirm complete=True: mark all targets, not only easy examples')
        if p['role'] not in ['train','calibration']:raise ValueError('Use train or calibration role')
        if any(overlaps(b,q['bounds']) for q in patches[:j]):raise ValueError('Training/calibration patches must not overlap')
        (tr if p['role']=='train' else va).append((3,tuple(b)))
        for bb in p['boxes']:
            if len(bb)!=4 or not (b[0]<=bb[0]<bb[2]<=b[2] and b[1]<=bb[1]<bb[3]<=b[3]):raise ValueError('Aperture boxes must lie within their labeled patch')
            boxes.append(bb)
    if not tr or not va:raise ValueError('Provide at least one complete training patch and one separate calibration patch')
    if not any(p['boxes'] for p in patches if p['role']=='calibration'):raise ValueError('Calibration needs at least one target to select point F1')
    b=np.asarray(boxes,np.float32).reshape(-1,4);xy=(b[:,:2]+b[:,2:])/2;rr=np.sqrt(np.prod(b[:,2:]-b[:,:2],axis=1)/np.pi)
    gt=np.column_stack([xy,rr]);cases={3:dict(im=rgb,boxes=b,gt=gt)}
    if a.replay_source:
        cases[a.replay_source]=prepare_cases(PROJECT/'data')[a.replay_source];tr.insert(0,(a.replay_source,(0,0,640,1024)))
    # Positive and negative appearances need to be represented in training;
    # this guard only checks at least one marked positive, not target quality.
    if not any(any(p['boxes']) for p in patches if p['role']=='train'):raise ValueError('At least one training target is required')
    tag='user_'+a.tag
    if (ROOT/'runs'/tag).exists():raise ValueError('This tag already has a run; choose a fresh tag for new annotations')
    save(ROOT/f'results/{tag}_annotations.json',dict(image=a.image,annotation_spec=spec,base_model=a.base_model,weight_origin=a.weight_origin,notes='User-supplied complete patches; no independent accuracy evaluation is performed here'))
    run=train(tag,cases,tr,va,a.base_model,a.epochs,True);chosen=calibrate(tag,run,cases,va)
    config=dict(model=str(ROOT/f'models/{tag}.pt'),threshold=chosen['threshold'],weight_origin=a.weight_origin,calibration_f1=chosen['f1'],scope='fitted to this image; calibration F1 is not independent performance')
    save(ROOT/f'models/{tag}.json',config);print(json.dumps(config,indent=2))
if __name__=='__main__':main()
