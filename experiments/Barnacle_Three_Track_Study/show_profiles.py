import sys,json
from pathlib import Path
import numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from profile_features import PROJECT,RADIAL
from barnacles.modern import prepare_cases
from barnacles.core import propose,sample,normalized_gray
R=Path(__file__).resolve().parent;c=prepare_cases(PROJECT/'data')[2];p,_=propose(c['im']);g=normalized_gray(c['im']);gt=c['gt']
valid=(gt[:,2]>=8)&(gt[:,2]<=17);candidates=gt[valid];positive=candidates[np.argmin(np.linalg.norm(candidates[:,:2]-[512,512],axis=1))]
d,j=cKDTree(gt[:,:2]).query(p[:,:2]);valid=(d>np.maximum(10,1.5*gt[j,2]))&(p[:,0]>80)&(p[:,0]<944)&(p[:,1]>80)&(p[:,1]<944)&(p[:,2]>7)
pp=p[valid];negative=pp[np.argmax(pp[:,3])]
fig,axs=plt.subplots(1,3,figsize=(14,4.5));fig.subplots_adjust(bottom=.19,top=.8,wspace=.28)
theta=np.linspace(0,2*np.pi,24,endpoint=False)
for ax,q,name,color in zip(axs[:2],[positive,negative],['Supplied labeled opening','Unmarked high-response candidate'],['#287ba0','#b55b38']):
 x,y=np.rint(q[:2]).astype(int);s=44;ax.imshow(c['im'][y-s:y+s,x-s:x+s]);ax.plot([s],[s],'+',color=color,markersize=12)
 for r in [.4,.85,1.45]:ax.plot(s+q[2]*r*np.cos(np.r_[theta,theta[0]]),s+q[2]*r*np.sin(np.r_[theta,theta[0]]),color=color,alpha=.8,linewidth=1)
 ax.set_title(name,fontsize=12);ax.set_xticks([]);ax.set_yticks([])
 z=sample(g,q[0]+q[2]*RADIAL[:,None]*np.cos(theta),q[1]+q[2]*RADIAL[:,None]*np.sin(theta))
 axs[2].plot(RADIAL,z.mean(1),'-o',color=color,label=name);axs[2].fill_between(RADIAL,np.quantile(z,.1,axis=1),np.quantile(z,.9,axis=1),color=color,alpha=.15)
axs[2].set_xlabel('Distance from candidate center / radius');axs[2].set_ylabel('Normalized image brightness');axs[2].legend(fontsize=8);axs[2].set_title('Keep the profile, not one average')
fig.suptitle('A broader descriptor of inward brightness changes',fontsize=17,fontweight='bold');fig.text(.5,.025,'Lines: angular means. Shading: 10th–90th brightness percentiles. The classifier also keeps sector profiles and radial differences.\n“Unmarked” does not establish an empty shell, species or live/dead status. These are measured image signals, not heights.',ha='center',fontsize=10)
fig.savefig(R/'Brightness_Profiles.png',dpi=150);print('Brightness_Profiles.png')
