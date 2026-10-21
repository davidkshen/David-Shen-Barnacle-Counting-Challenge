"""Soft, interpretable aperture-shape evidence. Inference never reads masks.

An area-equivalent radius r, minor/major ratio q, and superellipse exponent p
define a shape. p=2 is an ellipse; p=1 is an affine diamond. Shapes are only
image templates, not a species identification or fitted biological boundary.
"""
from dataclasses import dataclass, asdict
import math
import cv2
import numpy as np


@dataclass(frozen=True)
class EllipseConfig:
    ratios: tuple = (.5, .7, 1.)
    powers: tuple = (1., 2.)
    multipliers: tuple = (.75, 1., 1.4)
    tolerances: tuple = (.10, .25)
    angles: int = 24
    orientations: int = 6
    schema: int = 1


def axes(radius, ratio, power=2.):
    """Semiaxes with enclosed area pi*r^2, including for diamonds."""
    if not 0 < ratio <= 1 or power <= 0:
        raise ValueError('ratio must lie in (0,1], and power must be positive')
    unit_area = 4*math.gamma(1+1/power)**2/math.gamma(1+2/power)
    scale = math.sqrt(math.pi/unit_area)
    return radius*scale/math.sqrt(ratio), radius*scale*math.sqrt(ratio)


def remap(gray, x, y):
    """Bilinear image sampling (OpenCV quantizes fractions to 1/32 pixel)."""
    return cv2.remap(gray, np.ascontiguousarray(x, dtype=np.float32),
                     np.ascontiguousarray(y, dtype=np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT_101)


def coordinates(proposals, ratio, power, multiplier, angle, n=24):
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    u = np.sign(np.cos(t))*np.abs(np.cos(t))**(2/power)
    v = np.sign(np.sin(t))*np.abs(np.sin(t))**(2/power)
    a,b = axes(proposals[:,2,None]*multiplier, ratio, power)
    dx = a*u*np.cos(angle)-b*v*np.sin(angle)
    dy = a*u*np.sin(angle)+b*v*np.cos(angle)
    return dx.astype(np.float32), dy.astype(np.float32)


def seam_evidence(gray, proposals, ratio, power, multiplier, orientation):
    """Best soft seam and longest run, allowing small gaps and any length.

    Fractions refer to each tested image-plane central chord, not the major
    axis diameter. No mandatory seam length is imposed. A one-sample gap may
    be bridged; raw support remains separately available to the scorer.
    """
    n=len(proposals); t=np.linspace(-.85,.85,25)[None,:]
    a,b=axes(proposals[:,2,None]*multiplier,ratio,power)
    ca,sa=np.cos(orientation)[:,None],np.sin(orientation)[:,None]
    x,y=proposals[:,0,None],proposals[:,1,None]
    best=np.full(n,-np.inf,np.float32); out=np.zeros((n,4),np.float32)
    for phi in np.linspace(0,np.pi,6,endpoint=False):
        chord=(abs(np.cos(phi))**power+abs(np.sin(phi))**power)**(-1/power)
        ux=(a*np.cos(phi)*ca-b*np.sin(phi)*sa)*chord
        uy=(a*np.cos(phi)*sa+b*np.sin(phi)*ca)*chord
        length=np.maximum(np.hypot(ux,uy),1)
        xx=x+t*ux; yy=y+t*uy
        width=np.maximum(.75,.14*proposals[:,2,None]*multiplier)
        nx,ny=-uy/length*width,ux/length*width
        delta=.5*(remap(gray,xx+nx,yy+ny)+remap(gray,xx-nx,yy-ny))-remap(gray,xx,yy)
        strength=delta.mean(1)
        keep=strength>best; best=np.maximum(best,strength)
        support=delta>.2; filled=support.copy()
        filled[:,1:-1] |= support[:,:-2] & support[:,2:]
        run=np.zeros(n,int); longest=run.copy()
        for j in range(25):
            run=np.where(filled[:,j],run+1,0); longest=np.maximum(longest,run)
        values=np.column_stack((strength,support.mean(1),np.maximum(0,longest-1)*.85/24,
                                np.maximum(delta,0).mean(1)))
        out[keep]=values[keep]
    return out


def extract(gray, proposals, config=EllipseConfig(), progress=None):
    """Orientation-pool coherent rim+interior evidence for each shape bank.

    Choose orientation by rim mean + .3*lower quartile; measure the seam for
    that SAME shape, never combining a ring from one shape with another seam.
    Tolerance searches response centers; finite-width samples are r*(1+off
    +/- .12), so tolerance is not a literal contour-fitting certificate.
    """
    n=len(proposals); banks=[]; specs=[]
    x,y=proposals[:,0,None],proposals[:,1,None]
    for power in config.powers:
      for q in config.ratios:
       for mult in config.multipliers:
        for tol in config.tolerances:
            best=np.full(n,-np.inf,np.float32)
            values=np.zeros((n,8),np.float32); orient=np.zeros(n,np.float32)
            angles=[0.] if q==1 and power==2 else np.linspace(0,np.pi,config.orientations,endpoint=False)
            for angle in angles:
                dx,dy=coordinates(proposals,q,power,mult,angle,config.angles)
                edge=np.full(dx.shape,-np.inf,np.float32)
                for off in (-tol,0.,tol):
                    outer=remap(gray,x+(1+off+.12)*dx,y+(1+off+.12)*dy)
                    inner=remap(gray,x+(1+off-.12)*dx,y+(1+off-.12)*dy)
                    edge=np.maximum(edge,outer-inner)
                q25=np.quantile(edge,.25,axis=1)
                rank=edge.mean(1)+.3*q25
                keep=rank>best
                if np.any(keep):
                    rim=remap(gray,x+dx,y+dy)
                    center=remap(gray,x+.45*dx,y+.45*dy)
                    vv=np.column_stack((edge.mean(1),q25,(edge>.15).mean(1),(edge>.4).mean(1),
                        edge.std(1),(edge[:,:config.angles//2]*edge[:,config.angles//2:]>0).mean(1),
                        (rim-center).mean(1),rim.std(1)))
                    values[keep]=vv[keep]; orient[keep]=angle
                    best=np.maximum(best,rank)
            seam=seam_evidence(gray,proposals,q,power,mult,orient)
            banks.append(np.column_stack((values,seam)))
            specs.append(dict(power=power,ratio=q,multiplier=mult,tolerance=tol))
            if progress: progress(len(specs),len(config.powers)*len(config.ratios)*len(config.multipliers)*len(config.tolerances))
    return np.stack(banks,axis=1).astype(np.float32),specs


def classifier_features(bank, proposals, specs, family='all'):
    mask=np.array([family=='all' or (s['power']==2 and (family=='ellipses' or s['ratio']==1)) for s in specs])
    return np.column_stack((bank[:,mask].reshape(len(proposals),-1),np.log(proposals[:,2]))).astype(np.float32)


def soft_rule_scores(bank, proposals, specs, family='all', tolerance=.1,
                     seam_weight=.4,size_weight=.4, radius_median=7.,radius_log_sigma=.5):
    mask=np.array([s['tolerance']==tolerance and
        (family=='all' or (s['power']==2 and (family=='ellipses' or s['ratio']==1))) for s in specs])
    ids=np.flatnonzero(mask); v=bank[:,ids]
    # No hard coverage, radius, or seam gate. Low angular support lowers score.
    rim=np.maximum(v[:,:,0]+.3*v[:,:,1],0)*np.sqrt(v[:,:,2])
    seam=np.clip(v[:,:,8],0,2)*v[:,:,9]
    radii=proposals[:,2,None]*np.array([specs[i]['multiplier'] for i in ids])[None,:]
    prior=np.minimum(((np.log(radii/radius_median))/radius_log_sigma)**2,4)
    scores=rim+seam_weight*seam-size_weight*prior
    which=scores.argmax(1)
    return scores[np.arange(len(proposals)),which],ids[which]


def augment(rgb, proposals, name):
    """Deterministic nuisance views. Caller supplies TRAINING-only proposals.

    Rotations/flips are largely redundant with orientation pooling and are not
    counted as independent captures. Scaling transforms positions AND radii.
    """
    p=proposals.copy(); arr=rgb.astype(np.float32)/255
    if name.startswith('gamma'):
        gamma=float(name.split('_')[1]); arr=arr**gamma
    elif name=='smooth_shadow':
        h,w=rgb.shape[:2]; yy,xx=np.mgrid[:h,:w]
        shade=.6+.4*(1-np.exp(-((xx-.25*w)**2+(yy-.65*h)**2)/(2*(.35*w)**2)))
        arr*=shade[:,:,None]
    elif name=='blur_06':
        arr=cv2.GaussianBlur(arr,(0,0),.6)
    elif name=='noise_008':
        arr+=np.random.default_rng(42).normal(0,.008,arr.shape).astype(np.float32)
    elif name.startswith('scale'):
        scale=float(name.split('_')[1]); nh,nw=np.rint(np.array(rgb.shape[:2])*scale).astype(int)
        arr=cv2.resize(arr,(nw,nh),interpolation=cv2.INTER_LINEAR)
        sx,sy=nw/rgb.shape[1],nh/rgb.shape[0]
        p[:,0]=(p[:,0]+.5)*sx-.5; p[:,1]=(p[:,1]+.5)*sy-.5
        p[:,2]*=math.sqrt(sx*sy)
    else:
        raise ValueError(name)
    return np.rint(np.clip(arr,0,1)*255).astype(np.uint8),p


def split_membership(proposals):
    # 288px center gap. Largest feature+normalization support <130 original
    # pixels even for 0.85x zoom. Expanded train/validation views do not touch.
    return proposals[:,0]<368,proposals[:,0]>=656

