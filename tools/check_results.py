"""Check the full run, finite results, and key cross-project invariants."""
import json
import math
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from run import PROJECTS

def finite(value):
    if isinstance(value,float): assert math.isfinite(value), value
    elif isinstance(value,dict):
        for item in value.values(): finite(item)
    elif isinstance(value,list):
        for item in value: finite(item)

report=json.loads((ROOT/'generated/validation.json').read_text())
assert {r['project'] for r in report['runs']}==set(PROJECTS), 'Run python run.py all first'
assert all(r['status']=='passed' for r in report['runs'])
results={}
for name in PROJECTS:
    results[name]=json.loads((ROOT/'generated'/name/'metrics.json').read_text())
    assert results[name], name
    finite(results[name])
s=results['mie1624-salary-statistics']
assert s['retained_rows']==11748 and s['bootstrap_resamples']==10000
low,high=s['bootstrap_95_percent_ci']
assert low<s['mean_difference_usd']<high
assert 0<=results['aps1070-satellite-anomalies']['test_auc']<=1
p=results['aps1070-sales-pca']['reconstruction']
assert p[-1]['test_standardized_mse']<1e-20, 'Full PCA basis should reconstruct the input'
a=results['aps1070-aileron-regression']
assert a['gradient_descent_test_rmse']/a['least_squares_test_rmse']<1.1
assert results['che1148-retail-sql']['summary_rows']==10
assert results['che1148-toronto-housing']['test_months']==12
print(f'Validated finite outputs and analytical invariants for {len(PROJECTS)} projects.')
