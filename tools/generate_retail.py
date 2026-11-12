"""Regenerate the explicitly synthetic retail demo (no real customer input)."""
from pathlib import Path
import numpy as np
import pandas as pd

def generate():
    rng=np.random.default_rng(42);n=1600
    base=rng.lognormal(3.4,.8,n);active=rng.integers(1,15,n);d=pd.DataFrame()
    for year in range(2011,2016):
        count=rng.poisson(active*(1+.08*(year-2011)))+1
        average=base*rng.lognormal(0,.2,n)
        d[f'ann_txn_amt_cnt_{year}']=count
        d[f'ann_txn_amt_ave_{year}']=average
        d[f'ann_txn_amt_sum_{year}']=count*average
    score=-2.4+.6*np.log1p(d.ann_txn_amt_sum_2015/100)+.04*active
    d['response']=rng.binomial(1,1/(1+np.exp(-score)))
    for col in [c for c in d if c!='response']:
        d.loc[rng.random(n)<.03,col]=np.nan
    return d

if __name__=='__main__':
    dest=Path(__file__).resolve().parents[1]/'data/retail-annual.csv'
    generate().to_csv(dest,index=False,lineterminator='\n')
    print(f'Generated {dest.name}: 1,600 fictional customers.')
