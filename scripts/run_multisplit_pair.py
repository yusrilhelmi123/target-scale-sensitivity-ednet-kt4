#!/usr/bin/env python
"""Run the four learner-disjoint splits with five training seeds per split.

For each split/seed pair, train the APTQ-inspired scaled-RTG policy first,
then train fixed-CQL with alpha matched to that run's final adaptive alpha.
"""
import argparse
import exp_sensitivity as E

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--transitions-csv', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--splits', nargs='*', type=int, default=[20260728,1234,5678,9012])
    p.add_argument('--seeds', nargs='*', type=int, default=[2026,2027,2028,2029,2030])
    a=p.parse_args()
    for sp in a.splits:
        for seed in a.seeds:
            rec=E.job_ms(a.transitions_csv,a.out_dir,sp,seed,'aptq')
            E.job_ms(a.transitions_csv,a.out_dir,sp,seed,'cql_matched',rec['final_alpha'])
if __name__=='__main__':
    main()
