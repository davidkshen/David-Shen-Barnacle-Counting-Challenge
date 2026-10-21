"""Tiled appearance detector with contour-derived aperture boxes.

This module does not redefine the legacy reference centers or evaluation.
Labels are decoded only by `aperture_boxes`; inference only needs RGB/model.
"""
from pathlib import Path
import cv2
import numpy as np
from scipy import ndimage as ndi
from PIL import Image
from .core import rectify,read_rgb,load_reference,metrics


def aperture_boxes(mask_path,matrix,size=1024):
    rgba=np.asarray(Image.open(mask_path).convert('RGBA'))
    rgb=rgba[:,:,:3].astype(np.int16)
    stroke=(rgb[:,:,2]>rgb[:,:,0]+40)&(rgb[:,:,2]>rgb[:,:,1])&(rgba[:,:,3]>=32)
    holes=ndi.binary_fill_holes(stroke)&~stroke; lab,_=ndi.label(holes)
    areas=np.bincount(lab.ravel());ids=np.flatnonzero(areas>=6);ids=ids[ids!=0]
    objects=ndi.find_objects(lab);boxes=[];centers=[]
    for idx in ids:
        sl=objects[idx-1]; yy,xx=np.where(lab[sl]==idx)
        pts=np.column_stack((xx+sl[1].start,yy+sl[0].start)).astype(np.float32)
        center=pts.mean(0)
        c=cv2.perspectiveTransform(center[None,None],matrix)[0,0]
        if np.any(c<0) or np.any(c>=size):continue
        pts=cv2.perspectiveTransform(pts[:,None],matrix)[:,0]
        # Approximate half contour-stroke correction, matching the legacy
        # effective-radius convention. These boxes describe apertures, not shells.
        lo=pts.min(0)-1.5;hi=pts.max(0)+1.5
        boxes.append([max(0,lo[0]),max(0,lo[1]),min(size,hi[0]),min(size,hi[1])])
        centers.append(c)
    return np.asarray(boxes,np.float32),np.asarray(centers,np.float32)


def prepare_cases(data_dir):
    import json
    data_dir=Path(data_dir);frames=json.loads((data_dir/'frames.json').read_text());cases={}
    for k in (1,2):
        im,matrix=rectify(read_rgb(data_dir/f'img{k}.png'),frames[f'img{k}.png'])
        gt=load_reference(data_dir/f'mask{k}.png',matrix)
        boxes,centers=aperture_boxes(data_dir/f'mask{k}.png',matrix)
        np.testing.assert_allclose(centers,gt[:,:2],atol=1e-3)
        cases[k]=dict(im=im,gt=gt,boxes=boxes)
    return cases


def starts(low,high,tile=256,stride=192):
    if high-low<=tile:return [low]
    v=list(range(low,high-tile+1,stride))
    if v[-1]!=high-tile:v.append(high-tile)
    return v


def windows(bounds=(0,0,1024,1024),tile=256,stride=192):
    x0,y0,x1,y1=bounds;xs=starts(x0,x1,tile,stride);ys=starts(y0,y1,tile,stride)
    xx=[min(x+tile,x1) for x in xs];yy=[min(y+tile,y1) for y in ys]
    cx=np.array([(a+b)/2 for a,b in zip(xs,xx)])
    cy=np.array([(a+b)/2 for a,b in zip(ys,yy)])
    xc=[x0]+((cx[:-1]+cx[1:])/2).tolist()+[x1]
    yc=[y0]+((cy[:-1]+cy[1:])/2).tolist()+[y1]
    out=[]
    for j,y in enumerate(ys):
        for i,x in enumerate(xs):
            out.append(dict(crop=(x,y,xx[i],yy[j]),core=(xc[i],yc[j],xc[i+1],yc[j+1])))
    return out


def tile_predict(model,rgb,bounds=(0,0,1024,1024),imgsz=512,conf=.001):
    """Context-overlap tiles; each point belongs to one core; merge duplicates."""
    ws=windows(bounds);imgs=[rgb[w['crop'][1]:w['crop'][3],w['crop'][0]:w['crop'][2],::-1].copy() for w in ws]
    results=model.predict(imgs,imgsz=imgsz,conf=conf,iou=.5,max_det=2000,device='cpu',
        batch=4,verbose=False,save=False)
    pp=[];bb=[];scores=[]
    for w,r in zip(ws,results):
        if r.boxes is None or not len(r.boxes):continue
        b=r.boxes.xyxy.cpu().numpy();s=r.boxes.conf.cpu().numpy()
        b+=np.array([w['crop'][0],w['crop'][1],w['crop'][0],w['crop'][1]])
        xy=(b[:,:2]+b[:,2:])/2;c=w['core']
        keep=(xy[:,0]>=c[0])&(xy[:,0]<c[2])&(xy[:,1]>=c[1])&(xy[:,1]<c[3])
        bb.extend(b[keep]);scores.extend(s[keep])
    if not scores:return np.empty((0,4),np.float32),np.empty(0,np.float32),np.empty((0,4),np.float32)
    b=np.asarray(bb,np.float32);s=np.asarray(scores,np.float32)
    xywh=np.column_stack((b[:,:2],b[:,2:]-b[:,:2]))
    ids=cv2.dnn.NMSBoxes(xywh.tolist(),s.tolist(),conf,.5)
    ids=np.asarray(ids,dtype=int).ravel();b=b[ids];s=s[ids]
    xy=(b[:,:2]+b[:,2:])/2
    radius=np.sqrt(np.prod(b[:,2:]-b[:,:2],axis=1)/np.pi)
    p=np.column_stack((xy,radius,s)).astype(np.float32)
    return p,s,b


def choose_confidence(pred,scores,reference):
    thresholds=np.unique(np.r_[np.linspace(.01,.95,48),.001,.005,
        np.quantile(scores,np.linspace(.1,.99,20)) if len(scores) else []])
    rows=[dict(threshold=float(t),**metrics(pred[scores>=t],reference)) for t in thresholds]
    chosen=max(rows,key=lambda r:(r['f1'],r['precision']))
    return chosen,rows

