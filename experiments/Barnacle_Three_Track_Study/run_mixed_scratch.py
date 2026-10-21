"""Additional learnability diagnostic: both photo styles, new held regions."""
from run_detectors import ROOT,PROJECT,train,calibrate,evaluate,save
from barnacles.modern import prepare_cases
import torch
import ultralytics.utils.torch_utils as torch_utils
torch_utils.NUM_THREADS=5
torch.set_num_threads(5)
cases=prepare_cases(PROJECT/'data');tag='scratch_mixed_styles'
tr=[(k,(0,0,384,1024)) for k in (1,2)]
va=[(k,(512,0,704,1024)) for k in (1,2)]
ev=[(k,(832,0,1024,1024)) for k in (1,2)]
run=train(tag,cases,tr,va,'yolo11n.yaml',100)
locked=calibrate(tag,run,cases,va);rows=evaluate(tag,locked,cases,ev)
tp=sum(r['tp'] for r in rows);fp=sum(r['fp'] for r in rows);fn=sum(r['fn'] for r in rows)
save(ROOT/'results/mixed_scratch_summary.json',dict(validation_f1=locked['f1'],threshold=locked['threshold'],evaluations=rows,pooled_f1=2*tp/(2*tp+fp+fn),scope='held regions of both represented photos; not new-photo generalization'))
