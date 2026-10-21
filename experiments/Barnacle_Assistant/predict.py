"""Suggest centers and generate an offline human review tool. No masks needed."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import cv2
import joblib
import numpy as np
from PIL import Image
from barnacles.core import Config, read_rgb, rectify, propose, features, gradient_patches, select_detections
from barnacles.model import patch_stability_scores
from barnacles.reviewer import create_review


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',type=Path,required=True)
    p.add_argument('--model',type=Path,default=Path('results/model_from_img1.joblib'))
    p.add_argument('--frames',type=Path,default=Path('data/frames.json'))
    p.add_argument('--corners',help='JSON list of four inside corners, in original pixels: TL,TR,BR,BL')
    p.add_argument('--output',type=Path,default=Path('review_exports'))
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);cv2.setNumThreads(2)
    frames=json.loads(a.frames.read_text()) if a.frames.exists() else {}
    corners=json.loads(a.corners) if a.corners else frames.get(a.image.name)
    if corners is None:
        status=dict(status='abstain',reason='No calibrated frame. Supply --corners after inspecting the inside quadrat.')
        (a.output/'status.json').write_text(json.dumps(status,indent=2));print(status['reason']);return
    model=joblib.load(a.model);cfg=Config(**model['config'])
    image,matrix=rectify(read_rgb(a.image),corners,cfg.size)
    proposals,gray=propose(image,cfg)
    if len(proposals):
        X,_=features(gray,proposals);H=gradient_patches(gray,proposals)
        scores,views=patch_stability_scores(image,proposals,model['classifier'],X,H)
        ids=select_detections(proposals,scores,model['threshold'])
        pred=np.column_stack((proposals[ids],scores[ids],views[ids].std(1)))
    else:
        pred=np.empty((0,6))
    orig=(cv2.perspectiveTransform(pred[:,None,:2].astype(np.float32),np.linalg.inv(matrix))[:,0]
          if len(pred) else np.empty((0,2)))
    np.savetxt(a.output/'predictions.csv',np.column_stack((pred,orig)),delimiter=',',
        header='rectified_x,rectified_y,radius,ring_response,stability_score,view_score_sd,original_x,original_y',comments='')
    Image.fromarray(image).save(a.output/'rectified.jpg',quality=94)
    create_review(image,pred,a.output/'review.html',a.image.name+' / unverified suggestions')
    status=dict(status='human_review_required',suggestions=len(pred),corners=corners,
        model_source_image=model['source_image'],note='No supplied labels used. Stability score is not a calibrated probability.')
    (a.output/'status.json').write_text(json.dumps(status,indent=2))
    print(f'{len(pred)} unverified suggestions. Open {a.output/"review.html"} in your browser.')


if __name__=='__main__':main()
