"""RGB-only tiled inference with selected pilot weights and confidence."""
import argparse,json,sys
from pathlib import Path
import numpy as np,cv2
from PIL import Image
PROJECT=Path(__file__).resolve().parent/'project'
if not PROJECT.exists():PROJECT=Path(__file__).resolve().parents[1]/'recovered/project/barnacle_assistant'
sys.path.insert(0,str(PROJECT))
from barnacles.core import rectify,read_rgb
from barnacles.modern import tile_predict
from ultralytics import YOLO
import torch

def main():
    p=argparse.ArgumentParser();p.add_argument('--image',required=True);p.add_argument('--model',required=True);p.add_argument('--threshold',required=True,type=float);p.add_argument('--corners');p.add_argument('--output',default='prediction');a=p.parse_args()
    torch.set_num_threads(4)
    if a.corners:corners=json.loads(a.corners)
    else:
        frames=json.loads((PROJECT/'data/frames.json').read_text());corners=frames.get(Path(a.image).name)
        if corners is None:raise ValueError('Supply --corners as JSON [[TLx,TLy],[TRx,TRy],[BRx,BRy],[BLx,BLy]]')
    rgb,_=rectify(read_rgb(a.image),corners);pred,score,boxes=tile_predict(YOLO(a.model),rgb);keep=score>=a.threshold;pred=pred[keep];boxes=boxes[keep]
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);np.savetxt(out/'predictions.csv',pred,delimiter=',',header='x,y,radius,confidence',comments='')
    overlay=rgb.copy()
    for b in boxes:
        x0,y0,x1,y1=np.rint(b).astype(int);cv2.rectangle(overlay,(x0,y0),(x1,y1),(255,180,30),1)
    Image.fromarray(overlay).save(out/'overlay.png');(out/'count.json').write_text(json.dumps(dict(predictions=len(pred),threshold=a.threshold,model=a.model,status='unverified aperture detections; not a living-animal count'),indent=2))
    print(json.dumps(dict(predictions=len(pred),output=str(out))))
if __name__=='__main__':main()
