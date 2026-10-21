"""Inspectable ring/long-dark-line evidence. No annotation access or learning.

Radius means the visible inner opening/rim, not the whole animal's basal shell.
Ring tolerance is radial search of an IMAGE EDGE; it is not a fitted contour
certificate. Unrelated edges can provide apparent angular support.
"""
from dataclasses import dataclass, asdict
import numpy as np
from .core import sample, suppress


@dataclass(frozen=True)
class GeometricConfig:
    radial_tolerance: float = .05
    min_ring_coverage: float = .75
    min_line_diameter_fraction: float = .70
    ring_contrast: float = .20
    line_contrast: float = .20
    radius_multipliers: tuple = (.8, 1., 1.2)
    ring_angles: int = 48
    line_angles: int = 24
    line_samples: int = 41
    min_radius: float | None = None
    max_radius: float | None = None


def longest_span(support, step_in_diameters):
    """Span between endpoints of the longest contiguous supported sample run."""
    current = np.zeros(support.shape[:-1], dtype=np.int16)
    longest = current.copy()
    for j in range(support.shape[-1]):
        current = np.where(support[..., j], current+1, 0)
        longest = np.maximum(longest, current)
    return np.maximum(longest-1, 0)*step_in_diameters


def ring_strength(gray, proposals, radii, tolerance=.05, angles=48):
    """Per-angle strongest outward dark-to-bright edge in [(1-t)r,(1+t)r]."""
    theta = np.linspace(0, 2*np.pi, angles, endpoint=False)
    ux, uy = np.cos(theta), np.sin(theta)
    x, y = proposals[:,0,None], proposals[:,1,None]
    r = radii[:,None]; width = np.maximum(.5, .12*r)
    best = np.full((len(radii),angles), -np.inf, dtype=np.float32)
    for offset in np.linspace(-tolerance,tolerance,5):
        rr = r*(1+offset)
        outer = sample(gray, x+(rr+width)*ux, y+(rr+width)*uy)
        inner = sample(gray, x+(rr-width)*ux, y+(rr-width)*uy)
        best = np.maximum(best, outer-inner)
    return best


def line_spans(gray, proposals, radii, contrasts=(.1,.2,.35), angles=24, samples=41):
    """Straight dark line, any of 24 orientations, offsets -0.15r/0/+0.15r.

    Scan t in [-0.9r,0.9r]. The flank comparison must hold CONTIGUOUSLY.
    A 0.7-diameter segment is 1.4r long; nearby isolated dark pixels cannot
    accumulate to pass. Scores are in locally normalized luminance units.
    """
    best = np.zeros((len(proposals),len(contrasts)),dtype=np.float32)
    t = np.linspace(-.9,.9,samples)[None,:]
    x,y,r=proposals[:,0,None],proposals[:,1,None],radii[:,None]
    width=np.maximum(.75,.18*r)
    for a in np.linspace(0,np.pi,angles,endpoint=False):
        ux,uy=np.cos(a),np.sin(a)
        for off in (-.15,0.,.15):
            xx=x+r*t*ux-r*off*uy; yy=y+r*t*uy+r*off*ux
            center=sample(gray,xx,yy)
            delta=.5*(sample(gray,xx-width*uy,yy+width*ux)+sample(gray,xx+width*uy,yy-width*ux))-center
            for j,c in enumerate(contrasts):
                span=longest_span(delta>=c, .9/(samples-1))
                best[:,j]=np.maximum(best[:,j],span)
    return best


def evidence(gray, proposals, tolerance=.05, contrasts=(.1,.2,.35), multipliers=(.8,1.,1.2), batch=2048,
             ring_angles=48,line_angles=24,line_samples=41):
    """Return angular ring support and longest straight-line fractions.

    Safe pruning: line features are skipped only when no tested ring contrast
    can meet the minimum 50% coverage used in the experiment. Do not later
    use this cache with ring coverage <50%.
    """
    n=len(proposals); rc=np.zeros((n,len(multipliers),len(contrasts)),np.float32)
    ls=rc.copy(); strength=np.zeros((n,len(multipliers)),np.float32)
    for k,m in enumerate(multipliers):
        for start in range(0,n,batch):
            p=proposals[start:start+batch]; r=p[:,2]*m
            v=ring_strength(gray,p,r,tolerance,ring_angles)
            coverage=np.stack([(v>=c).mean(1) for c in contrasts],axis=1)
            rc[start:start+len(p),k]=coverage
            strength[start:start+len(p),k]=np.maximum(v,0).mean(1)
            keep=coverage[:,0]>=.5
            if np.any(keep):
                vals=line_spans(gray,p[keep],r[keep],contrasts,line_angles,line_samples)
                ls[start+np.flatnonzero(keep),k]=vals
    return dict(ring_coverage=rc,line_fraction=ls,ring_strength=strength,
                contrasts=np.array(contrasts),multipliers=np.array(multipliers),
                radial_tolerance=np.array(tolerance))


def detect(proposals, ev, config=GeometricConfig(), ring_only=False):
    if config.min_ring_coverage < .5:
        raise ValueError('Cached line evidence is valid only for ring coverage >=0.5')
    if not np.isclose(float(ev['radial_tolerance']),config.radial_tolerance):
        raise ValueError('Evidence has a different ring tolerance')
    ri=np.flatnonzero(np.isclose(ev['contrasts'],config.ring_contrast))
    li=np.flatnonzero(np.isclose(ev['contrasts'],config.line_contrast))
    if len(ri)!=1 or len(li)!=1: raise ValueError('Contrast is absent from this evidence cache')
    ring=ev['ring_coverage'][:,:,ri[0]]/config.min_ring_coverage
    scores=ring if ring_only else np.minimum(ring,ev['line_fraction'][:,:,li[0]]/config.min_line_diameter_fraction)
    radii=proposals[:,2,None]*ev['multipliers'][None,:]
    if config.min_radius is not None:scores=np.where(radii>=config.min_radius,scores,-np.inf)
    if config.max_radius is not None:scores=np.where(radii<=config.max_radius,scores,-np.inf)
    which=np.argmax(scores+1e-5*ev['ring_strength'],axis=1)
    score=scores[np.arange(len(proposals)),which]
    adjusted=proposals.copy();adjusted[:,2]*=ev['multipliers'][which]
    ids=np.flatnonzero(score>=1-1e-6)
    if len(ids):
        rank=score[ids]+1e-5*ev['ring_strength'][ids,which[ids]]
        ids=ids[suppress(adjusted[ids,:2],rank,distance=4.,radii=adjusted[ids,2])]
    return adjusted[ids], score[ids], ids
