"""Train pretrained tiled YOLO; source validation chooses checkpoint/confidence.

Full counterpart-photo transfer remains exploratory because these photos have
been used previously. This experiment is not limited to the earlier hand rules.
"""
import argparse,json,os,sys,time,hashlib
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from barnacles.modern import prepare_cases,windows,tile_predict,choose_confidence
from barnacles.core import metrics

ROOT=Path(__file__).resolve().parents[1]


def save(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def build_dataset(cases,folder,train_regions,val_regions,fraction=1.,seed=42):
    folder.mkdir(parents=True,exist_ok=True);manifest=[]
    for split,regions in [('train',train_regions),('val',val_regions)]:
        (folder/f'images/{split}').mkdir(parents=True,exist_ok=True)
        (folder/f'labels/{split}').mkdir(parents=True,exist_ok=True)
        tiles=[]
        for k,bounds in regions:
            for w in windows(bounds):tiles.append((k,w['crop']))
        original_tiles=list(tiles)
        if split=='train' and fraction<1:
            # Nested, per-photo selections keep capture-condition coverage.
            # Repeat selected tiles to hold optimization steps constant; repeats
            # remain the same training observations, never new independent data.
            selected=[]
            for k in sorted(set(k for k,_ in tiles)):
                subset=[t for t in tiles if t[0]==k]
                order=np.random.default_rng(seed+k).permutation(len(subset))
                selected.extend(subset[i] for i in order[:max(1,round(len(subset)*fraction))])
            tiles=[selected[j%len(selected)] for j in range(len(original_tiles))]
        for j,(k,(x0,y0,x1,y1)) in enumerate(tiles):
            name=f'img{k}_{j:03d}';c=cases[k];im=c['im'][y0:y1,x0:x1]
            Image.fromarray(im).save(folder/f'images/{split}/{name}.png')
            b=c['boxes'].copy();xy=(b[:,:2]+b[:,2:])/2
            keep=(xy[:,0]>=x0)&(xy[:,0]<x1)&(xy[:,1]>=y0)&(xy[:,1]<y1)
            b=b[keep];b-=np.array([x0,y0,x0,y0]);h,w=im.shape[:2]
            b[:,[0,2]]=np.clip(b[:,[0,2]],0,w);b[:,[1,3]]=np.clip(b[:,[1,3]],0,h)
            labels=[]
            for bb in b:
                cx,cy=(bb[:2]+bb[2:])/2;ww,hh=bb[2:]-bb[:2]
                if ww>1 and hh>1:labels.append(f'0 {cx/w:.8f} {cy/h:.8f} {ww/w:.8f} {hh/h:.8f}')
            (folder/f'labels/{split}/{name}.txt').write_text('\n'.join(labels)+'\n')
            manifest.append(dict(name=name,split=split,photo=k,crop=[x0,y0,x1,y1],labels=len(labels),
                source_observation=f'{k}:{x0}:{y0}:{x1}:{y1}'))
    config=folder/'data.yaml';config.write_text(f'path: {folder.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: annotated_aperture\n')
    save(folder/'manifest.json',manifest)
    return config


def train(data,tag,epochs=80,model='yolo11n.pt',imgsz=512,patience=100):
    from ultralytics import YOLO
    import torch
    torch.set_num_threads(6)
    yolo=YOLO(model)
    yolo.train(data=str(data),epochs=epochs,imgsz=imgsz,batch=8,device='cpu',workers=0,
        optimizer='AdamW',lr0=.001,lrf=.1,cos_lr=True,warmup_epochs=3,weight_decay=.0005,
        patience=patience,seed=42,deterministic=True,pretrained=True,
        degrees=180,translate=.1,scale=.3,fliplr=.5,flipud=.5,
        hsv_h=.015,hsv_s=.35,hsv_v=.35,mosaic=.3,close_mosaic=15,mixup=0.,
        max_det=2000,save=True,save_period=20,plots=False,verbose=False,
        cache='ram',project=str(ROOT/'modern_analysis/runs'),name=tag,exist_ok=True)
    return ROOT/'modern_analysis/runs'/tag


def calibrate_and_evaluate(run,cases,validation_regions,evaluation_regions,tag,imgsz=512):
    from ultralytics import YOLO
    candidates=sorted((run/'weights').glob('*.pt'));selection=[];best=None
    for checkpoint in candidates:
        model=YOLO(checkpoint);pred=[];scores=[];gts=[]
        # Offsets prevent matching between independent photos in pooled validation.
        for j,(k,bounds) in enumerate(validation_regions):
            p,s,_=tile_predict(model,cases[k]['im'],bounds,imgsz)
            gt=cases[k]['gt'];x0,y0,x1,y1=bounds
            gt=gt[(gt[:,0]>=x0)&(gt[:,0]<x1)&(gt[:,1]>=y0)&(gt[:,1]<y1)].copy()
            p[:,0]+=j*2048;gt[:,0]+=j*2048
            pred.append(p);scores.append(s);gts.append(gt)
        pp=np.concatenate(pred);ss=np.concatenate(scores);gg=np.concatenate(gts)
        chosen,curve=choose_confidence(pp,ss,gg)
        record=dict(checkpoint=str(checkpoint.relative_to(ROOT)),**chosen)
        selection.append(record)
        if best is None or (chosen['f1'],chosen['precision'])>(best['f1'],best['precision']):best=record
    save(ROOT/f'modern_analysis/results/{tag}_source_selection.json',selection)
    save(ROOT/f'modern_analysis/results/{tag}_locked.json',best)
    print('SOURCE LOCKED',tag,best,flush=True)
    model=YOLO(ROOT/best['checkpoint']);rows=[]
    for k,bounds in evaluation_regions:
        p,s,b=tile_predict(model,cases[k]['im'],bounds,imgsz)
        ids=s>=best['threshold'];pred=p[ids];boxes=b[ids];gt=cases[k]['gt'];x0,y0,x1,y1=bounds
        gt=gt[(gt[:,0]>=x0)&(gt[:,0]<x1)&(gt[:,1]>=y0)&(gt[:,1]<y1)]
        stats=metrics(pred,gt)
        rows.append(dict(tag=tag,evaluation_photo=k,bounds=list(bounds),threshold=best['threshold'],
            source_validation_f1=best['f1'],**stats,strict_center_f1=metrics(pred,gt,tolerance=.5)['f1']))
        np.savetxt(ROOT/f'modern_analysis/results/{tag}_on{k}_predictions.csv',pred,delimiter=',',
            header='x,y,predicted_aperture_radius,confidence',comments='')
        np.savetxt(ROOT/f'modern_analysis/results/{tag}_on{k}_boxes.csv',boxes,delimiter=',',header='x0,y0,x1,y1',comments='')
        print('EVALUATION',rows[-1],flush=True)
    save(ROOT/f'modern_analysis/results/{tag}_metrics.json',rows)
    return rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['transfer','mixed'],default='transfer')
    ap.add_argument('--source',type=int,choices=[1,2],default=1);ap.add_argument('--epochs',type=int,default=80)
    ap.add_argument('--fraction',type=float,default=1.);ap.add_argument('--imgsz',type=int,default=512)
    ap.add_argument('--train-width',type=int,default=640)
    ap.add_argument('--patience',type=int,default=100)
    ap.add_argument('--model',default='yolo11n.pt');ap.add_argument('--evaluate-only',action='store_true')
    args=ap.parse_args();cases=prepare_cases(ROOT/'data')
    (ROOT/'modern_analysis/results').mkdir(parents=True,exist_ok=True)
    if args.mode=='transfer':
        suffix='' if args.train_width==640 else f'_w{args.train_width}'
        tag=f'transfer_from{args.source}_{Path(args.model).stem}{suffix}'
        tr=[(args.source,(0,0,args.train_width,1024))];va=[(args.source,(768,0,1024,1024))];ev=[(3-args.source,(0,0,1024,1024))]
    else:
        tag=f'mixed_{args.fraction:g}_{Path(args.model).stem}'
        tr=[(k,(0,0,384,1024)) for k in (1,2)]
        va=[(k,(512,0,704,1024)) for k in (1,2)]
        ev=[(k,(832,0,1024,1024)) for k in (1,2)]
    save(ROOT/f'modern_analysis/results/{tag}_protocol.json',dict(train_regions=tr,validation_regions=va,
        evaluation_regions=ev,args=vars(args),model_initialization='COCO pretrained',
        targets='decoded aperture boxes expanded 1.5px; legacy centers unchanged',
        checkpoint_selection='source validation point F1 only',
        prior_photo_reuse='exploratory; no untouched independent field test'))
    folder=ROOT/'modern_analysis/datasets'/tag
    data=build_dataset(cases,folder,tr,va,args.fraction)
    run=ROOT/'modern_analysis/runs'/tag
    if not args.evaluate_only:run=train(data,tag,args.epochs,args.model,args.imgsz,args.patience)
    calibrate_and_evaluate(run,cases,va,ev,tag,args.imgsz)


if __name__=='__main__':main()
