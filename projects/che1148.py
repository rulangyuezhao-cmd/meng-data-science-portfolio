"""CHE1148: data quality, interpretation, drift and time-series analysis."""
import sqlite3
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.metrics import roc_auc_score, f1_score, mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from .common import DATA,SEED,output,finish,plt

def retail_quality():
    name='che1148-retail-quality';out=output(name)
    df=pd.read_csv(DATA/'retail-annual.csv');X=df.drop(columns='response');y=df.response
    assert y.isin([0,1]).all() and not df.duplicated().all()
    a,b,ya,yb=train_test_split(X,y,test_size=.25,stratify=y,random_state=SEED)
    model=make_pipeline(SimpleImputer(strategy='median'),RandomForestClassifier(n_estimators=100,max_depth=8,min_samples_leaf=5,class_weight='balanced',random_state=SEED,n_jobs=2))
    model.fit(a,ya);prob=model.predict_proba(b)[:,1]
    imp=permutation_importance(model,b,yb,n_repeats=3,random_state=SEED,scoring='roc_auc',n_jobs=1)
    importance=pd.Series(imp.importances_mean,index=X.columns).sort_values(ascending=False)
    importance.to_csv(out/'permutation-importance.csv',header=['mean_auc_decrease'])
    top=importance.index[0]
    PartialDependenceDisplay.from_estimator(model,b,features=[top],kind='average');plt.tight_layout();plt.savefig(out/'partial-dependence.png',dpi=160)
    # Deliberate perturbation, not a claim that real production drift occurred.
    shifted=b.copy();shifted[top]=shifted[top]+a[top].std()*1.5
    edges=np.histogram_bin_edges(pd.concat([a[top],b[top],shifted[top]]).dropna(),bins=20)
    hist=lambda s:np.histogram(s.dropna(),bins=edges)[0]+1e-9
    before=float(jensenshannon(hist(a[top]),hist(b[top])))
    after=float(jensenshannon(hist(a[top]),hist(shifted[top])))
    missing=X.isna().mean().sort_values(ascending=False);missing.to_csv(out/'missing-fractions.csv')
    return finish(name,{'train_customers':len(a),'test_customers':len(b),'features':X.shape[1],
       'test_auc':roc_auc_score(yb,prob),'test_f1':f1_score(yb,prob>=.5),'most_important_feature':top,
       'js_distance_held_out':before,'js_distance_simulated_shift':after,
       'scope':'Synthetic customer data generated for this public adaptation; metrics are demonstration results, not original course results. Partial dependence shows model response, not causal effects.'})

def retail_sql():
    name='che1148-retail-sql';out=output(name)
    df=pd.read_csv(DATA/'retail-annual.csv').reset_index(names='customer_key')
    cols=[c for c in df if c.startswith('ann_txn_amt_sum_')]
    long=df.melt(id_vars=['customer_key','response'],value_vars=cols,var_name='feature',value_name='annual_spend')
    long['year']=long.feature.str.extract(r'(\d{4})$').astype(int)
    with sqlite3.connect(':memory:') as conn:
        long[['customer_key','response','year','annual_spend']].to_sql('customer_year',conn,index=False)
        query=(DATA.parent/'courses/che1148/annual_metrics.sql').read_text()
        result=pd.read_sql_query(query,conn)
        assert result.customer_count.sum()==long.annual_spend.notna().sum()
    result.to_csv(out/'annual-kpis.csv',index=False)
    return finish(name,{'input_customers':len(df),'customer_year_rows':len(long),'summary_rows':len(result),'engine':'SQLite','query':'courses/che1148/annual_metrics.sql'})

def covid_visualization():
    name='che1148-country-visualization';out=output(name)
    df=pd.read_csv(DATA/'covid-country-totals.csv',parse_dates=['Date'])
    latest=df.sort_values('Date').groupby('Countries',as_index=False).tail(1).copy()
    latest=latest[(latest.Population>0)&(latest.Cases>0)&(latest.Deaths>=0)]
    latest['cases_per_million']=latest.Cases/latest.Population*1e6
    latest['deaths_per_million']=latest.Deaths/latest.Population*1e6
    latest.to_csv(out/'country-rate-comparison.csv',index=False)
    plt.figure(figsize=(8,5))
    for continent,part in latest.groupby('Continent'):
        plt.scatter(part.cases_per_million,part.deaths_per_million,label=continent,alpha=.7)
    plt.xscale('log');plt.yscale('symlog',linthresh=1);plt.xlabel('Reported cases per million');plt.ylabel('Reported deaths per million');plt.legend(fontsize=8)
    plt.tight_layout();plt.savefig(out/'country-rates.png',dpi=160)
    return finish(name,{'countries':len(latest),'latest_date':str(latest.Date.max().date()),
      'scope':'Historical country-level comparisons. Reporting and population differences limit comparisons; this is not routing or GIS analysis.'})

def housing_forecast():
    name='che1148-toronto-housing';out=output(name)
    df=pd.read_csv(DATA/'toronto-housing.csv',parse_dates=['date']).sort_values('date')
    y=df.set_index('date').price.asfreq('MS')
    assert y.notna().all() and len(y)>48
    train,test=y.iloc[:-12],y.iloc[-12:]
    # Last-year seasonal baseline and models fitted only to observations before the cutoff.
    # Scale dollars to millions for numerical optimization, then restore original units.
    hw=ExponentialSmoothing(train/1e6,trend='add',seasonal='add',seasonal_periods=12,initialization_method='estimated').fit()
    assert hw.mle_retvals['success'], 'Holt-Winters optimization did not converge'
    forecasts={'seasonal_naive':train.iloc[-12:].to_numpy(),
      'holt_winters':(hw.forecast(12)*1e6).to_numpy(),
      'sarima':SARIMAX(train,order=(1,1,1),seasonal_order=(0,1,1,12),enforce_stationarity=False,enforce_invertibility=False).fit(disp=False,maxiter=100).forecast(12).to_numpy()}
    results={k:{'mae':mean_absolute_error(test,p),'rmse':mean_squared_error(test,p)**.5} for k,p in forecasts.items()}
    pd.DataFrame({'actual':test,**forecasts}).to_csv(out/'held-out-forecast.csv')
    plt.figure(figsize=(9,4));plt.plot(train.iloc[-36:],label='Train');plt.plot(test,label='Held-out actual')
    for k,p in forecasts.items():plt.plot(test.index,p,label=k)
    plt.legend(fontsize=8);plt.ylabel('Average home price');plt.tight_layout();plt.savefig(out/'forecast.png',dpi=160)
    return finish(name,{'train_months':len(train),'test_months':12,'cutoff':str(train.index[-1].date()),'models':results,
      'scope':'Historical forecasting exercise; original notebook also explores Prophet and additional diagnostics.'})

PROJECTS={'che1148-retail-quality':retail_quality,'che1148-retail-sql':retail_sql,'che1148-country-visualization':covid_visualization,'che1148-toronto-housing':housing_forecast}
