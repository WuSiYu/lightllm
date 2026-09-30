#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser(); p.add_argument('inputs',nargs='+'); p.add_argument('--out',required=True); args=p.parse_args()
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    plt.figure(figsize=(8,4.5))
    for fn in args.inputs:
        rows=[json.loads(l) for l in open(fn) if l.strip()]; ts=[r['timestamp']/1000 for r in rows];
        bins={}
        for r,t in zip(rows,ts): bins.setdefault(int(t//10),[]).append(r['input_length']>4000)
        x=sorted(bins); y=[sum(bins[k])/len(bins[k]) for k in x]
        plt.plot([k*10/60 for k in x],y,label=Path(fn).stem)
    plt.xlabel('trace time (minutes)'); plt.ylabel('fraction input length > 4000'); plt.ylim(0,1); plt.grid(alpha=.25); plt.legend(); plt.tight_layout(); plt.savefig(out,dpi=220); print(out)
if __name__=='__main__': main()
