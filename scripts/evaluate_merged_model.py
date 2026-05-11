"""
Merged 5-Class Model Evaluation on 344-image Test Set
=======================================================
Evaluates the merged v7 model on the same frozen 344-image test set.
Uses the MERGED test labels (5 classes, roadwork dropped).

Classes:
  0 = car
  1 = large_vehicle  (bus + truck)
  2 = traffic_light
  3 = cyclist        (bicycle + motorcycle)
  4 = person

Results saved to: data/results_merged/

Author: Sumit Malviya (W24041293)
"""

import os, re, time, csv
import numpy as np
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO

# ============================================================
# CONFIGURATION
# ============================================================
# MODEL_PATH  = "runs/detect/newcastle_v6_merged/weights/best.pt"
# DATASET_DIR = Path("data/Newcastle-Traffic-Detection.v6_merged")
# TEST_DIR    = DATASET_DIR / "test"
# RESULTS_DIR = Path("data/results_v6_merged")
# RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH  = "runs/detect/newcastle_v9_5class/weights/best.pt"
DATASET_DIR = Path("data/Newcastle-Traffic-Detection.v6_merged")
TEST_DIR    = DATASET_DIR / "test"
RESULTS_DIR = Path("data/results_v9_5class")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CONF   = 0.25
DEVICE = "cpu"
IMGSZ  = 960

RF_NAMES = {0:"car", 1:"large_vehicle", 2:"traffic_light", 3:"cyclist", 4:"person"}
NUM_CLASSES = len(RF_NAMES)
IOU_THRESHOLDS = np.arange(0.5, 1.0, 0.05)

# ============================================================
# HELPERS
# ============================================================
def compute_iou(b1, b2):
    x1=max(b1[0]-b1[2]/2,b2[0]-b2[2]/2); y1=max(b1[1]-b1[3]/2,b2[1]-b2[3]/2)
    x2=min(b1[0]+b1[2]/2,b2[0]+b2[2]/2); y2=min(b1[1]+b1[3]/2,b2[1]+b2[3]/2)
    inter=max(0,x2-x1)*max(0,y2-y1)
    union=b1[2]*b1[3]+b2[2]*b2[3]-inter
    return inter/union if union>0 else 0

def greedy_match(preds, gts, iou_thr=0.5):
    if not preds or not gts:
        return [], list(range(len(preds))), list(range(len(gts)))
    M=np.zeros((len(preds),len(gts)))
    for i,p in enumerate(preds):
        for j,g in enumerate(gts):
            if p['class']==g['class']: M[i,j]=compute_iou(p['box'],g['box'])
    matches,mp,mg=[],set(),set()
    while True:
        if M.size==0 or M.max()<iou_thr: break
        pi,gi=np.unravel_index(M.argmax(),M.shape); pi,gi=int(pi),int(gi)
        matches.append((pi,gi,M[pi,gi])); mp.add(pi); mg.add(gi)
        M[pi,:]=0; M[:,gi]=0
    return matches,[i for i in range(len(preds)) if i not in mp],[i for i in range(len(gts)) if i not in mg]

def compute_ap(confs_tps, total_gt):
    if not confs_tps or total_gt==0: return 0.0
    sd=sorted(confs_tps,key=lambda x:x[0],reverse=True)
    tp=fp=0; ps,rs=[],[]
    for c,is_tp in sd:
        if is_tp: tp+=1
        else: fp+=1
        ps.append(tp/(tp+fp)); rs.append(tp/total_gt)
    return sum(max([p for p,r in zip(ps,rs) if r>=t],default=0) for t in np.arange(0,1.1,0.1))/11.0

def load_gt(path):
    boxes=[]
    if not os.path.exists(path): return boxes
    for line in open(path):
        p=line.strip().split()
        if len(p)>=5: boxes.append({'class':int(p[0]),'box':[float(x) for x in p[1:5]]})
    return boxes

def parse_hour(filename):
    m=re.search(r'(\d{8})_(\d{6})',filename)
    if m:
        h=int(m.group(2)[:2])
        return h if 0<=h<=23 else None
    return None

# ============================================================
# LOAD TEST DATA
# ============================================================
print(f"\n{'='*65}")
print(f"  MERGED 5-CLASS EVALUATION")
print(f"  Model:    {MODEL_PATH}")
print(f"  Test set: {TEST_DIR}")
print(f"  Classes:  {list(RF_NAMES.values())}")
print(f"{'='*65}\n")

test_data=[]
for img_path in sorted((TEST_DIR/'images').glob('*')):
    if img_path.suffix.lower() not in {'.jpg','.jpeg','.png'}: continue
    parts=img_path.stem.split('__')
    camera=parts[0] if len(parts)>=2 else img_path.stem.split('_')[0]
    hour=parse_hour(img_path.name)
    period='day' if hour is not None and 7<=hour<18 else 'night'
    test_data.append({'image':str(img_path),
                      'label':str(TEST_DIR/'labels'/f'{img_path.stem}.txt'),
                      'camera':camera,'period':period})

gt_counts=defaultdict(int)
for d in test_data:
    for g in load_gt(d['label']): gt_counts[g['class']]+=1

print(f"  Test images : {len(test_data)}")
print(f"  GT objects  : {sum(gt_counts.values())}")
for cid in sorted(RF_NAMES): print(f"    {RF_NAMES[cid]:15s}: {gt_counts[cid]}")

# ============================================================
# EVALUATE
# ============================================================
print(f"\n  Loading model: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

class_stats=defaultdict(lambda:{'TP':0,'FP':0,'FN':0})
period_stats=defaultdict(lambda:defaultdict(lambda:{'TP':0,'FP':0,'FN':0}))
per_class_confs=defaultdict(list)
per_class_strict={round(t,2):defaultdict(list) for t in IOU_THRESHOLDS}

t0=time.time()
for idx,item in enumerate(test_data):
    gts=load_gt(item['label'])
    results=model.predict(source=item['image'],conf=CONF,verbose=False,device=DEVICE,imgsz=IMGSZ)
    preds=[{'class':int(b.cls[0]),'box':b.xywhn[0].tolist(),'conf':float(b.conf[0])}
           for b in results[0].boxes if int(b.cls[0]) in RF_NAMES]

    matches,up,ug=greedy_match(preds,gts,0.5)
    for pi,gi,_ in matches:
        c=preds[pi]['class']; class_stats[c]['TP']+=1
        period_stats[item['period']][c]['TP']+=1
        per_class_confs[c].append((preds[pi]['conf'],True))
    for pi in up:
        c=preds[pi]['class']; class_stats[c]['FP']+=1
        period_stats[item['period']][c]['FP']+=1
        per_class_confs[c].append((preds[pi]['conf'],False))
    for gi in ug:
        c=gts[gi]['class']; class_stats[c]['FN']+=1
        period_stats[item['period']][c]['FN']+=1
    for iou_thr in IOU_THRESHOLDS:
        k=round(iou_thr,2); mt,_,_=greedy_match(preds,gts,iou_thr)
        matched={pi for pi,_,_ in mt}
        for pi,gi,_ in mt: per_class_strict[k][preds[pi]['class']].append((preds[pi]['conf'],True))
        for pi in range(len(preds)):
            if pi not in matched: per_class_strict[k][preds[pi]['class']].append((preds[pi]['conf'],False))
    if (idx+1)%50==0: print(f"  [{idx+1}/{len(test_data)}]")

elapsed=time.time()-t0
print(f"  Done in {elapsed:.0f}s")

# ============================================================
# COMPUTE METRICS
# ============================================================
per_class=[]; all_aps50=[]; all_aps5095=[]
for cid in sorted(RF_NAMES):
    s=class_stats[cid]; tp,fp,fn=s['TP'],s['FP'],s['FN']; gt=tp+fn
    p=tp/(tp+fp) if tp+fp>0 else 0; r=tp/(tp+fn) if tp+fn>0 else 0
    f1=2*p*r/(p+r) if p+r>0 else 0
    ap50=compute_ap(per_class_confs[cid],gt); all_aps50.append(ap50)
    ap5095=np.mean([compute_ap(per_class_strict[round(t,2)][cid],gt) for t in IOU_THRESHOLDS])
    all_aps5095.append(ap5095)
    per_class.append({'cid':cid,'name':RF_NAMES[cid],'GT':gt,'TP':tp,'FP':fp,'FN':fn,
                      'P':p,'R':r,'F1':f1,'AP50':ap50,'AP50_95':ap5095})

mAP50=np.mean(all_aps50); mAP5095=np.mean(all_aps5095)
ttp=sum(class_stats[c]['TP'] for c in RF_NAMES)
tfp=sum(class_stats[c]['FP'] for c in RF_NAMES)
tfn=sum(class_stats[c]['FN'] for c in RF_NAMES)
op=ttp/(ttp+tfp) if ttp+tfp>0 else 0
orr=ttp/(ttp+tfn) if ttp+tfn>0 else 0
of1=2*op*orr/(op+orr) if op+orr>0 else 0

# Day/night
day_night={}
for period in ['day','night']:
    ps=period_stats[period]
    ptp=sum(ps[c]['TP'] for c in RF_NAMES); pfp=sum(ps[c]['FP'] for c in RF_NAMES)
    pfn=sum(ps[c]['FN'] for c in RF_NAMES)
    pp=ptp/(ptp+pfp) if ptp+pfp>0 else 0; pr=ptp/(ptp+pfn) if ptp+pfn>0 else 0
    pf=2*pp*pr/(pp+pr) if pp+pr>0 else 0
    ni=sum(1 for d in test_data if d['period']==period)
    day_night[period]={'imgs':ni,'P':pp,'R':pr,'F1':pf}

# ============================================================
# PRINT RESULTS
# ============================================================
print(f"\n{'='*65}")
print(f"  RESULTS — 5-CLASS MERGED MODEL ON 344 TEST IMAGES")
print(f"{'='*65}")
print(f"\n  {'Class':<16} {'GT':>5} {'P':>7} {'R':>7} {'F1':>7} {'AP@0.5':>8} {'AP@5:95':>9}")
print(f"  {'-'*62}")
for pc in per_class:
    print(f"  {pc['name']:<16} {pc['GT']:5d} {pc['P']:6.1%} {pc['R']:6.1%} {pc['F1']:6.1%} {pc['AP50']:7.1%} {pc['AP50_95']:8.1%}")
tgt=ttp+tfn
print(f"\n  {'OVERALL':<16} {tgt:5d} {op:6.1%} {orr:6.1%} {of1:6.1%} {mAP50:7.1%} {mAP5095:8.1%}")
print(f"\n  mAP@0.5      = {mAP50:.4f} ({mAP50:.1%})")
print(f"  mAP@0.5:0.95 = {mAP5095:.4f} ({mAP5095:.1%})")
print(f"\n  Day vs Night:")
for period in ['day','night']:
    s=day_night[period]
    print(f"    {period:6s}: {s['imgs']:3d} imgs  P={s['P']:.1%}  R={s['R']:.1%}  F1={s['F1']:.1%}")

# ============================================================
# SAVE CSV
# ============================================================
with open(RESULTS_DIR/'overall.csv','w',newline='') as f:
    w=csv.writer(f)
    w.writerow(['metric','value'])
    w.writerow(['mAP50', round(mAP50,4)])
    w.writerow(['mAP50_95', round(mAP5095,4)])
    w.writerow(['precision', round(op,4)])
    w.writerow(['recall', round(orr,4)])
    w.writerow(['f1', round(of1,4)])

with open(RESULTS_DIR/'per_class.csv','w',newline='') as f:
    w=csv.writer(f)
    w.writerow(['class','GT','P','R','F1','AP50','AP50_95'])
    for pc in per_class:
        w.writerow([pc['name'],pc['GT'],round(pc['P'],4),round(pc['R'],4),
                    round(pc['F1'],4),round(pc['AP50'],4),round(pc['AP50_95'],4)])

with open(RESULTS_DIR/'day_night.csv','w',newline='') as f:
    w=csv.writer(f)
    w.writerow(['period','imgs','P','R','F1'])
    for period in ['day','night']:
        s=day_night[period]
        w.writerow([period,s['imgs'],round(s['P'],4),round(s['R'],4),round(s['F1'],4)])

print(f"\n  Results saved to {RESULTS_DIR}/")
print(f"\n{'='*65}")
print(f"  DONE")
print(f"{'='*65}")
