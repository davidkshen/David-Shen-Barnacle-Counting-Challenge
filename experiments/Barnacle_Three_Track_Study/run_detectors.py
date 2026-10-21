"""Matched initialization comparison, then per-photo supervised adaptation."""
import sys,json,hashlib,time,os
from pathlib import Path
import numpy as np
PROJECT=Path(__file__).resolve().parent/'project'
if not PROJECT.exists():PROJECT=Path(__file__).resolve().parents[1]/'recovered/project/barnacle_assistant'
sys.path.insert(0,str(PROJECT))
from modern_analysis.train_detector import build_dataset
from barnacles.modern import prepare_cases,tile_predict,choose_confidence
from barnacles.core import metrics
from ultralytics import YOLO
from ultralytics.utils.torch_utils import init_seeds
import torch
ROOT=Path(__file__).resolve().parent
for name in ['results','models','cache']:(ROOT/name).mkdir(exist_ok=True)
import ultralytics.utils.torch_utils as torch_utils
torch_utils.NUM_THREADS=int(os.environ.get('BARNACLE_CPU_THREADS','5'))
torch.set_num_threads(torch_utils.NUM_THREADS)

def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def gt_region(c,b):
    g=c['gt'];return g[(g[:,0]>=b[0])&(g[:,0]<b[2])&(g[:,1]>=b[1])&(g[:,1]<b[3])]
def train(tag,cases,tr,va,initial,epochs=100,adapt=False):
    folder=ROOT/'datasets'/tag;data=build_dataset(cases,folder,tr,va)
    run=ROOT/'runs'/tag
    if (run/'weights/last.pt').exists():
        import csv
        with (run/'results.csv').open() as f:done=list(csv.DictReader(f))
        if int(done[-1]['epoch'])>=epochs:return run
        raise RuntimeError(f'Incomplete run at {run}; resume it explicitly or move it before rerunning')
    init_seeds(42,deterministic=True);model=YOLO(initial)
    model.train(data=str(data),epochs=epochs,imgsz=512,batch=8,nbs=8,device='cpu',workers=0,
        optimizer='AdamW',lr0=.0003 if adapt else .001,lrf=.1,cos_lr=True,warmup_epochs=3,
        weight_decay=.0005,patience=epochs+1,seed=42,deterministic=True,
        pretrained=adapt or str(initial).endswith('.pt'),
        degrees=180,translate=.1,scale=.3,fliplr=.5,flipud=.5,hsv_h=.015,hsv_s=.35,hsv_v=.35,
        mosaic=0. if adapt else .3,close_mosaic=0 if adapt else 15,mixup=0.,max_det=2000,
        save=True,save_period=10 if adapt else 20,plots=False,verbose=False,cache='ram',
        project=str(ROOT/'runs'),name=tag,exist_ok=True)
    return run

def calibrate(tag,run,cases,regions):
    out=ROOT/f'results/{tag}_selection.json'
    rows=[]
    for ck in sorted((run/'weights').glob('*.pt')):
        model=YOLO(ck);pp=[];ss=[];gg=[]
        for j,(k,b) in enumerate(regions):
            p,s,_=tile_predict(model,cases[k]['im'],b);g=gt_region(cases[k],b).copy();p[:,0]+=j*2048;g[:,0]+=j*2048
            pp.append(p);ss.append(s);gg.append(g)
        chosen,_=choose_confidence(np.concatenate(pp),np.concatenate(ss),np.concatenate(gg))
        rows.append(dict(checkpoint=str(ck),**chosen))
    chosen=max(rows,key=lambda v:(v['f1'],v['precision']));save(out,dict(candidates=rows,chosen=chosen))
    selected=YOLO(chosen['checkpoint']);selected.save(ROOT/f'models/{tag}.pt');print('LOCKED',tag,chosen,flush=True)
    return chosen

def evaluate(tag,locked,cases,regions):
    model=YOLO(ROOT/f'models/{tag}.pt');rows=[]
    for k,b in regions:
        p,s,_=tile_predict(model,cases[k]['im'],b);p=p[s>=locked['threshold']];g=gt_region(cases[k],b)
        rows.append(dict(photo=k,bounds=b,**metrics(p,g),strict_center_f1=metrics(p,g,.5)['f1']))
        np.savetxt(ROOT/f'results/{tag}_on{k}_x{b[0]}.csv',p,delimiter=',',header='x,y,radius,confidence',comments='')
    save(ROOT/f'results/{tag}_evaluation.json',rows);print('EVALUATED',tag,rows,flush=True);return rows

def main():
    cases=prepare_cases(PROJECT/'data');summary=[];selected={}
    for source in (1,2):
        for initialization in ('scratch','pretrained'):
            tag=f'{initialization}_from{source}';tr=[(source,(0,0,640,1024))];va=[(source,(768,0,1024,1024))]
            initial='yolo11n.yaml' if initialization=='scratch' else str(PROJECT/'yolo11n.pt')
            run=train(tag,cases,tr,va,initial);locked=calibrate(tag,run,cases,va);selected[(source,initialization)]=locked
            rows=evaluate(tag,locked,cases,[(3-source,(0,0,1024,1024)),(3-source,(832,0,1024,1024))])
            summary.append(dict(tag=tag,source=source,target=3-source,initialization=initialization,
                epochs=100,validation_f1=locked['f1'],threshold=locked['threshold'],evaluations=rows))
            save(ROOT/'results/detector_summary.json',summary)
    hybrid=[]
    for source in (1,2):
        for initialization in ('scratch','pretrained'):
            target=3-source;base=selected[(source,initialization)];tag=f'{initialization}_adapt_to{target}'
            # Replay source tiles alongside the new fully labeled training tile.
            # The separate target calibration tile is included in annotation cost.
            tr=[(source,(0,0,640,1024)),(target,(0,0,256,256))];va=[(target,(512,384,768,640))]
            run=train(tag,cases,tr,va,str(ROOT/f'models/{initialization}_from{source}.pt'),30,True)
            locked=calibrate(tag,run,cases,va);rows=evaluate(tag,locked,cases,[(target,(832,0,1024,1024))])
            ntrain=len(gt_region(cases[target],tr[-1][1]));ncal=len(gt_region(cases[target],va[0][1]))
            # Threshold-only comparison isolates inexpensive local recalibration.
            model=YOLO(ROOT/f'models/{initialization}_from{source}.pt');p,s,_=tile_predict(model,cases[target]['im'],va[0][1]);chosen,_=choose_confidence(p,s,gt_region(cases[target],va[0][1]))
            evalmodel=ROOT/f'models/{initialization}_from{source}.pt';tt=YOLO(evalmodel);p,s,_=tile_predict(tt,cases[target]['im'],(832,0,1024,1024));threshold_stats=metrics(p[s>=chosen['threshold']],gt_region(cases[target],(832,0,1024,1024)))
            row=dict(tag=tag,initialization=initialization,source=source,target=target,training_labels=ntrain,calibration_labels=ncal,
                annotation_labels=ntrain+ncal,annotation_pixel_fraction=.125,threshold_only=threshold_stats,adapted=rows[0])
            hybrid.append(row);save(ROOT/'results/detector_hybrid.json',hybrid);print('HYBRID',row,flush=True)

if __name__=='__main__':main()
