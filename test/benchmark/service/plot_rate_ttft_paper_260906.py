#!/usr/bin/env python3
"""Build p99-TTFT vs offered-rate figures from completed sweep CSVs."""
import argparse, csv, glob, os
from pathlib import Path
import matplotlib.pyplot as plt

def load(path):
    with open(path) as f: return list(csv.DictReader(f))

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',default='_/flex_tp_paper_analysis'); p.add_argument('--out-dir',required=True); args=p.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    files=glob.glob(os.path.join(args.root,'**','rate_ttft_sweep.csv'),recursive=True)
    groups=[]
    for f in files:
        rows=load(f)
        if not rows or 'ttft_p99_s' not in rows[0] or 'request_rate' not in rows[0]: continue
        dataset=rows[0].get('dataset','unknown'); sched=sorted(set(r.get('scheduler','') for r in rows));
        if not any(s in sched for s in ('fixed_tp2','fixed_tp4','v12','v13','naive')): continue
        groups.append((dataset,Path(f).parent.name,f,rows))
    # Keep one representative per dataset: prefer the most recent path lexically.
    by={}
    for g in groups: by[g[0]]=g
    for dataset,name,f,rows in sorted(by.values()):
        plt.figure(figsize=(7,4.5))
        for s in sorted(set(r.get('scheduler','') for r in rows)):
            x=sorted((float(r['request_rate']),float(r['ttft_p99_s'])) for r in rows if r.get('scheduler')==s)
            if x: plt.plot([a for a,b in x],[b for a,b in x],marker='o',label=s)
        plt.xlabel('offered request rate (req/s)'); plt.ylabel('TTFT p99 (s)'); plt.title(f'{dataset} ({name})'); plt.grid(alpha=.25); plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(out/f'{dataset}.ttft_p99.png',dpi=180); plt.close()
    print(f'generated {len(by)} figures in {out}')
if __name__=='__main__': main()
