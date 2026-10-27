#!/usr/bin/env python3
"""OVCD 变化检测精度评价脚本"""
import numpy as np
from PIL import Image
import glob, os, sys

def evaluate(pred_dir, label_dir):
    labels = {os.path.basename(p): p for p in glob.glob(os.path.join(label_dir, "*.png"))}
    tp = fp = fn = tn = 0
    n = 0

    for p in glob.glob(os.path.join(pred_dir, "*.png")):
        name = os.path.basename(p)
        if name not in labels:
            continue
        n += 1
        pred = np.array(Image.open(p))
        # RGB 转二值：非黑即变化
        if pred.ndim == 3:
            pd = (pred.max(axis=2) > 0).astype(np.uint8).flatten()
        else:
            pd = (pred > 0).astype(np.uint8).flatten()
        gt = (np.array(Image.open(labels[name])) > 0).astype(np.uint8).flatten()

        tp += int((pd & gt).sum())
        fp += int((pd & (1 - gt)).sum())
        fn += int(((1 - pd) & gt).sum())
        tn += int(((1 - pd) & (1 - gt)).sum())

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    iou  = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    acc  = (tp + tn) / (tp + fp + fn + tn)

    print(f"匹配: {n} 对")
    print(f"F1={f1:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  IoU={iou:.4f}  Acc={acc:.4f}")
    print(f"TP={tp}  FP={fp}  FN={fn}  TN={tn}")


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        evaluate(sys.argv[1], sys.argv[2])
    else:
        # 默认：Proposed_multiclass 消失建筑
        PRED = "/media/lenovo/课题研究/博士小论文数据/开放词汇CD/sam3/sam3代码/results/SECOND/buildling_results/Proposed_multiclass"
        LABEL = "/media/lenovo/备份/nas/临时共享/开放词汇CD/test/labels/3_disappeared_building"
        print(f"预测: {PRED}\n标签: {LABEL}\n")
        evaluate(PRED, LABEL)
