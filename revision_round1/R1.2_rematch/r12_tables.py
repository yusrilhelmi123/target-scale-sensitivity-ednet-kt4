import sys, json, csv
from pathlib import Path
import numpy as np
ARC=Path(sys.argv[1]); DATA=sys.argv[2]; Q12=Path(sys.argv[3]); OUT=Path(sys.argv[4])
sys.path.insert(0, str(ARC/"scripts"))
import common as C, fastrl as F, clean_200k as K
sel=json.load(open(ARC/"results_clean_200k/selection.json")); rows=C.read_transitions(DATA)
train,val,test=K.split_fixed(rows,int(sel["split_seed"])); sidx=C.build_state_index(rows); NA=7; G=0.95
arr=C.to_arrays(train,sidx); te=C.to_arrays(test,sidx); ts,ta=te["s"],te["a"]
def mse(q): t=te["r"]+(1-te["done"])*G*q[te["ns"]].max(1); return float(np.mean((t-q[te["s"],te["a"]])**2))
am=lambda pi: float(np.mean(pi[ts]==ta))
def fq(pi,sd): qe=F.fqe_fit(pi,train,sidx,G,0.05,60,sd,NA,arr); return float(C.fqe_state_values(qe,test,pi,sidx).mean())
R={"old":[],"new":[],"apt":[]}
for sd in [2026,2027,2028,2029,2030]:
    z=np.load(ARC/f"results_clean_200k/seed_{sd}_q_tables.npz"); qn=np.load(Q12/f"q_cql_alphaEm1_seed{sd}.npy")
    for k,q in (("apt",z["aptq_norm_rtg"]),("old",z["cql_final_alpha_matched"]),("new",qn)):
        pi=q.argmax(1); R[k].append((mse(q),am(pi),fq(pi,sd)))
a={k:np.array(v) for k,v in R.items()}
for k in a: print(k,"mean",a[k].mean(0).round(4),"sd",a[k].std(0,ddof=1).round(4))
t=2.7764
for k in ("old","new"):
    d=a["apt"]-a[k]; m=d.mean(0); h=t*d.std(0,ddof=1)/np.sqrt(5)
    print("APTQ -",k,"d(MSE,AM,FQE):",[f"{m[i]:+.4f} [{m[i]-h[i]:+.4f}, {m[i]+h[i]:+.4f}]" for i in range(3)])
al=[json.load(open(ARC/"results_clean_200k/selection.json"))]  # unused
import csv
h={}
for sd in [2026,2027,2028,2029,2030]:
    hh=[float(r["alpha"]) for r in csv.DictReader(open(ARC/f"results_clean_200k/seed_{sd}_histories.csv")) if r["policy_id"]=="aptq_norm_rtg"]
    h[sd]=(hh[-1],hh[-2])
print("mean alpha_E %.4f, mean alpha_Em1 %.4f"%(np.mean([v[0] for v in h.values()]),np.mean([v[1] for v in h.values()])))
json.dump({k:v.tolist() for k,v in a.items()},open(OUT/"r12_table_rows.json","w"))
