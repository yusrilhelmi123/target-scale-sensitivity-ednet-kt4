#!/usr/bin/env python
"""Re-run the five-seed CQL fine grid on the primary validation split."""
import argparse,csv,os,sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

ALPHAS=[0.01,0.02,0.03,0.04,0.05,0.06,0.08,0.10]
SEEDS=[2026,2027,2028,2029,2030]

def amatch(q,rows,sidx):
    pol=C.greedy_policy(q)
    return float(np.mean([pol[sidx[r['state']]]==r['action'] for r in rows]))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--transitions-csv',required=True)
    p.add_argument('--out-csv',required=True)
    p.add_argument('--split-seed',type=int,default=20260728)
    a=p.parse_args()
    rows=C.read_transitions(a.transitions_csv); sidx=C.build_state_index(rows)
    tr,va,_=C.split_by_user_three_way(rows,0.15,0.15,a.split_seed)
    rec=[]
    for alpha in ALPHAS:
        ms=[]; am=[]
        for seed in SEEDS:
            q,_=C.train_q(tr,sidx,7,algorithm='cql',epochs=20,lr=0.08,gamma=0.95,cql_alpha=alpha,seed=seed)
            ms.append(C.bellman_mse(q,va,sidx,0.95)); am.append(amatch(q,va,sidx))
        rec.append((alpha,float(np.mean(ms)),float(np.mean(am))))
    os.makedirs(os.path.dirname(os.path.abspath(a.out_csv)),exist_ok=True)
    with open(a.out_csv,'w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['cql_alpha','validation_bellman_mse_mean_5seeds','validation_action_match_mean_5seeds']); w.writerows(rec)
if __name__=='__main__': main()
