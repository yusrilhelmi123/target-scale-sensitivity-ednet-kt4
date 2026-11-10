import json, glob, numpy as np, pandas as pd
from scipy import stats
rows=[json.loads(l) for f in sorted(glob.glob('e2_worlds_beta*.jsonl')) for l in open(f)]
print('worlds',len(rows), 'unique',len({r['world'] for r in rows}))
rng=np.random.default_rng(1)
def boot(x,B=4000,stat=np.mean):
    x=np.asarray(x); idx=rng.integers(0,len(x),(B,len(x))); s=np.array([stat(x[i]) for i in idx]); return float(np.quantile(s,.025)),float(np.quantile(s,.975))
P=['offline','cql','aptq','matched']
out={}
print('\n=== coverage summary')
for b in (0.5,1.5,3.0):
    R=[r for r in rows if r['beta']==b]
    print(f"beta={b}: frac pairs count<2 {np.mean([r['frac_pairs_lt2'] for r in R]):.3f}, zero {np.mean([r['frac_pairs_zero'] for r in R]):.3f}, true behaviour {np.mean([r['true_behaviour_vis'] for r in R]):.3f}, cql alpha median {np.median([r['cql_alpha'] for r in R])}, aptq alpha_E mean {np.mean([r['alpha_E'] for r in R]):.4f}, scale choices {pd.Series([r['apt_scale'] for r in R]).value_counts().to_dict()}")
print('\n=== H1: bias of learned own value (own - true), mean [95% boot over worlds]')
for b in (0.5,1.5,3.0):
    R=[r for r in rows if r['beta']==b]
    for p in P:
        x=[r[p]['own']-r[p]['true'] for r in R]; lo,hi=boot(x); print(f"beta={b} {p:8s} bias {np.mean(x):+.3f} [{lo:+.3f},{hi:+.3f}]  frac worlds >0: {np.mean(np.array(x)>0):.2f}   true value {np.mean([r[p]['true'] for r in R]):.3f}")
print('\n=== H2: bias of FQE and empirical-MDP estimate (est - true)')
for b in (0.5,1.5,3.0):
    R=[r for r in rows if r['beta']==b]
    for p in ['offline','cql','aptq']:
        for e in ('fqe','emp'):
            x=[r[p][e]-r[p]['true'] for r in R]; lo,hi=boot(x); print(f"beta={b} {p:8s} {e:4s} bias {np.mean(x):+.3f} [{lo:+.3f},{hi:+.3f}]  frac>0 {np.mean(np.array(x)>0):.2f}")
print('\n--- pairwise ranking accuracy among offline/cql/aptq (3 pairs per world) vs TRUE ranking')
pairs=[('offline','cql'),('offline','aptq'),('cql','aptq')]
for b in (0.5,1.5,3.0,'all'):
    R=[r for r in rows if b=='all' or r['beta']==b]
    acc={}
    for e in ('own','fqe','emp'):
        ok=[]
        for r in R:
            for a,c in pairs:
                t=np.sign(r[a]['true']-r[c]['true']); s=np.sign(r[a][e]-r[c][e]); ok.append(float(s==t))
        acc[e]=np.mean(ok)
    print(f"beta={b}: accuracy own {acc['own']:.3f}, fqe {acc['fqe']:.3f}, emp {acc['emp']:.3f}  (chance 0.5; n pairs {len(R)*3})")
print('\n=== H3: Spearman(action_match, true improvement over behaviour), pooled over 4 policies x worlds')
for b in (0.5,1.5,3.0,'all'):
    R=[r for r in rows if b=='all' or r['beta']==b]
    am=np.array([r[p]['action_match'] for r in R for p in P]); imp=np.array([r[p]['true']-r['true_behaviour_vis'] for r in R for p in P])
    rho,pv=stats.spearmanr(am,imp)
    # cluster bootstrap over worlds
    W=len(R); arr_am=np.array([[r[p]['action_match'] for p in P] for r in R]); arr_imp=np.array([[r[p]['true']-r['true_behaviour_vis'] for p in P] for r in R])
    bs=[]
    for _ in range(2000):
        i=rng.integers(0,W,W); bs.append(stats.spearmanr(arr_am[i].ravel(),arr_imp[i].ravel())[0])
    print(f"beta={b}: rho={rho:+.3f} [{np.quantile(bs,.025):+.3f},{np.quantile(bs,.975):+.3f}]  mean action match {am.mean():.3f}, mean true improvement {imp.mean():+.3f}, share of policies with true improvement>0: {np.mean(imp>0):.2f}")
print('\n--- per policy: mean action match / true improvement over behaviour')
for b in (0.5,1.5,3.0):
    R=[r for r in rows if r['beta']==b]
    print(f"beta={b}: "+'; '.join(f"{p}: am {np.mean([r[p]['action_match'] for r in R]):.3f}, impr {np.mean([r[p]['true']-r['true_behaviour_vis'] for r in R]):+.3f}" for p in P))
print('\n=== H4: paired true value aptq - matched (margin +/-0.10)')
for b in (0.5,1.5,3.0,'all'):
    R=[r for r in rows if b=='all' or r['beta']==b]
    d=np.array([r['aptq']['true']-r['matched']['true'] for r in R]); lo,hi=boot(d)
    dis=np.mean([r['disagree_apt_matched_test_w'] for r in R]); same=np.mean([r['aptq']['action_match']-r['matched']['action_match'] for r in R])
    print(f"beta={b}: diff {d.mean():+.4f} [{lo:+.4f},{hi:+.4f}] within margin: {'yes' if (lo>-0.1 and hi<0.1) else 'NO'}; |diff|max {np.abs(d).max():.3f}; worlds |d|>0.10: {(np.abs(d)>0.10).sum()}/{len(d)}; test-weighted disagreement {dis:.4f}; d action match {same:+.4f}")
print('\n=== extra: aptq vs best-tuned CQL true value (aptq - cql)')
for b in (0.5,1.5,3.0):
    R=[r for r in rows if r['beta']==b]; d=np.array([r['aptq']['true']-r['cql']['true'] for r in R]); lo,hi=boot(d); print(f"beta={b}: {d.mean():+.3f} [{lo:+.3f},{hi:+.3f}]")
pd.DataFrame([{ 'beta':r['beta'],'world':r['world'],'frac_lt2':r['frac_pairs_lt2'],'true_beh':r['true_behaviour_vis'],**{f'{p}_{k}':r[p][k] for p in P for k in ('own','fqe','emp','true','action_match')}, 'alpha_E':r['alpha_E'],'alpha_Em1':r['alpha_Em1']} for r in rows]).to_csv('e2_worlds_flat.csv',index=False)
