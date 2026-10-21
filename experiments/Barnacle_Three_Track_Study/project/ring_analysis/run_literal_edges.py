"""Literal pixel-edge annulus check, separate from finite-width ring filters.

For each radius, count angular bins containing Canny edge PIXELS with measured
distance in [(1-t)r,(1+t)r]. This is an exact distance gate on the chosen edge
map, not a guarantee about true physical shell boundaries or connectedness.
"""
from dataclasses import asdict,replace
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from barnacles.core import read_rgb,rectify,load_reference,propose,Config,metrics
from barnacles.geometric import GeometricConfig,detect


def angular_edge_coverage(edge_xy,centers,radii,tolerance,bins=48):
    tree=cKDTree(edge_xy);out=np.zeros((len(centers),radii.shape[1]),np.float32)
    for start in range(0,len(centers),512):
        nearby=tree.query_ball_point(centers[start:start+512],radii[start:start+512].max(1)*(1+tolerance))
        for j,ids in enumerate(nearby,start):
            if not ids:continue
            delta=edge_xy[ids]-centers[j];distance=np.linalg.norm(delta,axis=1)
            angle=np.floor(np.mod(np.arctan2(delta[:,1],delta[:,0]),2*np.pi)/(2*np.pi)*bins).astype(int)%bins
            for k,r in enumerate(radii[j]):
                keep=(distance>=r*(1-tolerance))&(distance<=r*(1+tolerance))
                out[j,k]=len(np.unique(angle[keep]))/bins
    return out


def main():
    frames=json.loads(Path('data/frames.json').read_text());rows=[]
    for k in (1,2):
        rgb,m=rectify(read_rgb(f'data/img{k}.png'),frames[f'img{k}.png'])
        p,g=propose(rgb,Config());gt=load_reference(f'data/mask{k}.png',m)
        # Fixed contrast mapping and thresholds, not chosen from evaluation labels.
        u=np.rint(255*(np.clip(g,-3,3)+3)/6).astype(np.uint8)
        edges=cv2.Canny(u,50,100,L2gradient=True)
        yy,xx=np.where(edges>0);edge_xy=np.column_stack((xx,yy)).astype(float)
        for tol in (.05,.20):
            ev=dict(np.load(f'ring_analysis/cache/img{k}_tol{tol}.npz'))
            radii=p[:,2,None]*ev['multipliers'][None,:]
            coverage=angular_edge_coverage(edge_xy,p[:,:2],radii,tol)
            ev['ring_coverage']=np.repeat(coverage[:,:,None],len(ev['contrasts']),axis=2)
            ev['ring_strength']=coverage
            # The filtered cache pruned line computation for low filtered-ring
            # support. Compute missing line features so this new ring gate does
            # not inherit that pruning. Only the fixed .2 contrast is required.
            from barnacles.geometric import line_spans
            old=np.load(f'ring_analysis/cache/img{k}_tol{tol}.npz')['ring_coverage']
            for ri in range(len(ev['multipliers'])):
                missing=(coverage[:,ri]>=.75)&(old[:,ri,0]<.5)
                if np.any(missing):
                    ev['line_fraction'][missing,ri,1]=line_spans(g,p[missing],radii[missing,ri],contrasts=(.2,))[:,0]
            cfg=replace(GeometricConfig(),radial_tolerance=tol)
            det,score,_=detect(p,ev,cfg)
            row=dict(image=k,profile=f'literal_edge_band_{int(tol*100)}pct_line70',
                config=asdict(cfg),canny_low=50,canny_high=100,angular_bins=48,**metrics(det,gt))
            rows.append(row);print(row,flush=True)
            np.savetxt(f'ring_analysis/results/img{k}_literal_{tol}.csv',np.column_stack((det,score)),
                delimiter=',',header='x,y,radius,proposal_ring_response,rule_score',comments='')
    Path('ring_analysis/results/literal_edge_metrics.json').write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':main()
