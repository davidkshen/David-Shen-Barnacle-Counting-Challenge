"""Fixed radial-sector descriptors; no learned outside weights or height claims."""
import sys
from pathlib import Path
import numpy as np
import cv2
PROJECT=Path(__file__).resolve().parent/'project'
if not PROJECT.exists():PROJECT=Path(__file__).resolve().parents[1]/'recovered/project/barnacle_assistant'
sys.path.insert(0,str(PROJECT))
from barnacles.core import sample,normalized_gray,gradient_patches

RADIAL=np.array([0.,.2,.4,.6,.85,1.1,1.45],np.float32)

def profile_features(rgb,proposals,with_hog=True):
    g=normalized_gray(rgb);x=proposals[:,0,None,None];y=proposals[:,1,None,None]
    theta=np.linspace(0,2*np.pi,24,endpoint=False)[None,None,:]
    radius=proposals[:,2,None,None]*RADIAL[None,:,None]
    result=[]
    for q in (1.,.65):
        orientations=[0.] if q==1 else [0.,np.pi/3,2*np.pi/3]
        views=[]
        for angle in orientations:
            xx=radius*np.cos(theta)/np.sqrt(q);yy=radius*np.sin(theta)*np.sqrt(q)
            u=xx*np.cos(angle)-yy*np.sin(angle);v=xx*np.sin(angle)+yy*np.cos(angle)
            z=sample(g,x+u,y+v)
            sectors=z.reshape(len(proposals),7,6,4).mean(3)
            # Sort COMPLETE sector profiles together; do not independently sort
            # radii, which would invent inward traces from different directions.
            contrast=sectors[:,-1]-sectors[:,0]
            order=np.argsort(contrast,axis=1)
            ordered=np.take_along_axis(sectors,order[:,None,:],axis=2)
            grad=np.diff(ordered,axis=1)
            quants=np.quantile(z,[.1,.5,.9],axis=2).transpose(1,2,0)
            # Broad interior brightness and narrow dark tails remain separate.
            # A potential drop/recovery is represented, never required.
            views.append(np.column_stack([ordered.reshape(len(z),-1),grad.reshape(len(z),-1),quants.reshape(len(z),-1)]))
        stack=np.stack(views,1)
        result.extend([stack.mean(1),stack.max(1)])
    result.append(np.log(np.maximum(proposals[:,2],.1))[:,None])
    profile=np.column_stack(result).astype(np.float32)
    return profile,np.column_stack([profile,gradient_patches(g,proposals)]).astype(np.float32) if with_hog else profile
