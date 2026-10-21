"""Extend the tiled detector to new, capture-group-separated framed photos.

Each manifest entry supplies image, mask, corners, capture_group and split.
Masks must use the same blue/alpha closed-contour format as the challenge.
The final test split is evaluated only after source validation selects settings.
"""
import argparse,json,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from barnacles.core import read_rgb,rectify,load_reference
from barnacles.modern import aperture_boxes
from modern_analysis.train_detector import build_dataset,train,calibrate_and_evaluate,ROOT,save


def validate_groups(entries):
    roles={}
    for row in entries:
        role=row['split'];group=row['capture_group']
        if role not in ('train','val','test'):raise ValueError('split must be train, val or test')
        if group in roles and roles[group]!=role:
            raise ValueError(f'Capture group {group!r} crosses dataset splits')
        roles[group]=role
    if set(roles.values())!={'train','val','test'}:
        raise ValueError('Provide capture groups for train, val and test')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',required=True,type=Path)
    ap.add_argument('--epochs',type=int,default=100)
    ap.add_argument('--initial-model',default='yolo11n.pt')
    args=ap.parse_args();entries=json.loads(args.manifest.read_text());validate_groups(entries)
    base=args.manifest.resolve().parent;cases={};regions={s:[] for s in ('train','val','test')}
    for k,row in enumerate(entries,1):
        image=(base/row['image']).resolve();mask=(base/row['mask']).resolve()
        im,matrix=rectify(read_rgb(image),row['corners'])
        gt=load_reference(mask,matrix);boxes,_=aperture_boxes(mask,matrix)
        cases[k]=dict(im=im,gt=gt,boxes=boxes)
        regions[row['split']].append((k,(0,0,1024,1024)))
    tag='custom_'+args.manifest.stem
    data=build_dataset(cases,ROOT/'modern_analysis/datasets'/tag,regions['train'],regions['val'])
    run=train(data,tag,args.epochs,args.initial_model,512,args.epochs)
    calibrate_and_evaluate(run,cases,regions['val'],regions['test'],tag)
    locked=json.loads((ROOT/f'modern_analysis/results/{tag}_locked.json').read_text())
    from ultralytics.utils.torch_utils import strip_optimizer
    folder=ROOT/'modern_analysis/models';folder.mkdir(parents=True,exist_ok=True)
    dst=folder/f'{tag}.pt';shutil.copyfile(ROOT/locked['checkpoint'],dst);strip_optimizer(dst)
    save(folder/f'{tag}.json',dict(model=str(dst.relative_to(ROOT)),threshold=locked['threshold'],imgsz=512,
        selected_using='capture-group-separated source validation',validation_f1=locked['f1'],training_run=tag))
    print('Prediction config:',folder/f'{tag}.json')


if __name__=='__main__':main()
