"""Prepare a random-tile count audit, or summarize independently filled counts.

Freeze the model before preparing an audit. Complete every selected tile,
including missed objects. Reusing this audit to fit/reselect the model invalidates
its role as an independent audit. Bounds remain approximate, not exact coverage.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/'project'))
if not (Path(__file__).resolve().parent/'project').exists():sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'recovered/project/barnacle_assistant'))
from barnacles.audit import tile_counts,random_tiles,corrected_count

def main():
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='mode',required=True)
    p=sub.add_parser('prepare');p.add_argument('--predictions',required=True);p.add_argument('--output',required=True);p.add_argument('--n',type=int,default=16);p.add_argument('--seed',type=int,default=42)
    s=sub.add_parser('summarize');s.add_argument('--audit',required=True)
    a=ap.parse_args()
    if a.mode=='prepare':
        if not 2<=a.n<=64:raise ValueError('Use 2 to 64 random tiles; a single tile cannot estimate sampling variance')
        xy=np.loadtxt(a.predictions,delimiter=',',skiprows=1,ndmin=2);counts=tile_counts(xy,grid=8);ids=random_tiles(8,a.n,a.seed)
        rows=[{'tile_id':int(i),'bounds':[int(i%8*128),int(i//8*128),int(i%8*128+128),int(i//8*128+128)],'model_count':int(counts[i]),'human_count':None} for i in ids]
        out={'model_counts':counts.tolist(),'sampling':'uniform without replacement AFTER model freeze','seed':a.seed,'sampled_ids':ids.tolist(),'audit':rows,'scope':'1024px rectified quadrat; center-based tile ownership'}
        Path(a.output).write_text(json.dumps(out,indent=2)+'\n');print(a.output)
    else:
        d=json.loads(Path(a.audit).read_text());expected=random_tiles(8,len(d['sampled_ids']),d['seed'])
        assert np.array_equal(expected,d['sampled_ids']),'Audit selection changed'
        assert [x['tile_id'] for x in d['audit']]==d['sampled_ids'],'Audit order changed'
        counts=[]
        for r in d['audit']:
            v=r['human_count']
            if not isinstance(v,int) or v<0:raise ValueError('Complete every selected tile with a nonnegative integer human count')
            counts.append(v)
        result=corrected_count(d['model_counts'],d['sampled_ids'],counts)
        print(json.dumps(result,indent=2,allow_nan=False))
if __name__=='__main__':main()
