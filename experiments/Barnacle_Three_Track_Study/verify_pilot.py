"""Focused numerical/split checks for the new descriptors and pilot protocol."""
import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from profile_features import profile_features,RADIAL
R=Path(__file__).resolve().parent
p=np.array([[128.,128.,7.5,.2],[100.,100.,10.,.3]],np.float32)
a,b=profile_features(np.full((256,256,3),128,np.uint8),p)
assert a.shape==(2,397) and b.shape==(2,525)
assert np.isfinite(b).all()
assert np.max(np.abs(a[:,:-1]))<1e-3
# A bright center and a dark center with the same radius remain distinguishable.
y,x=np.mgrid[:256,:256];d=np.hypot(x-128,y-128)
bright=np.where(d<4,220,np.where(d<8,40,160)).astype(np.uint8)
dark=np.where(d<4,40,np.where(d<8,220,160)).astype(np.uint8)
u,_=profile_features(np.repeat(bright[:,:,None],3,axis=2),p[:1])
v,_=profile_features(np.repeat(dark[:,:,None],3,axis=2),p[:1])
assert np.linalg.norm(u-v)>1
# Full sample support: largest ellipse sample plus finite normalization support.
support=17/np.sqrt(.65)*RADIAL.max()+48
assert 2*support<192
manifest=json.loads((R/'results/profile_hybrid_manifest.json').read_text())
for row in manifest:
 tr=row['train'];cal=row['calibration'];test=row['test']
 assert all(b[2]+2*support<cal[0] for b in tr)
 assert cal[2]+2*support<test[0]
# Source feature split also has enough guard space.
assert 368+2*support<656
protocol=json.loads((R/'results/protocol.json').read_text())
assert protocol['deep_adaptation_train'][2]<protocol['deep_adaptation_validation'][0]
assert protocol['deep_adaptation_validation'][2]<protocol['adaptation_test'][0]
print(json.dumps({'feature_dimensions':[397,525],'finite_support_radius_px':float(support),'constant_and_radial_structure_checks':True,'adaptation_manifests_disjoint_with_feature_support':len(manifest),'deep_pixel_regions_disjoint':True}))
