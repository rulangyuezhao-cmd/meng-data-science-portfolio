"""APS1052: fixed-input finance coursework adaptations, not investment advice."""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.api import VAR
from .common import DATA,SEED,output,finish,plt

def seasonal_sales():
    name='aps1052-seasonal-sales';out=output(name)
    df=pd.read_csv(DATA/'cocacola.csv');y=df.Sales.to_numpy(dtype=float)
    t=np.arange(len(y));quarter=t%4
    X=np.column_stack([t,*[(quarter==q).astype(int) for q in range(1,4)]])
    train=len(y)-8
    model=GridSearchCV(make_pipeline(StandardScaler(),Ridge()),{'ridge__alpha':[.01,.1,1.,10.]},cv=TimeSeriesSplit(4),scoring='neg_mean_absolute_error').fit(X[:train],np.log(y[:train]))
    pred=np.exp(model.predict(X[train:]));baseline=np.tile(y[train-4:train],2)
    pd.DataFrame({'actual':y[train:],'forecast':pred,'seasonal_naive':baseline}).to_csv(out/'forecasts.csv',index=False)
    return finish(name,{'train_quarters':train,'test_quarters':8,'alpha':model.best_params_['ridge__alpha'],'mae':mean_absolute_error(y[train:],pred),'baseline_mae':mean_absolute_error(y[train:],baseline),
      'scope':'Log-scale trend and quarterly seasonal model; back-transformation estimates a conditional median.'})

def yield_curve():
    name='aps1052-yield-curve';out=output(name)
    df=pd.read_csv(DATA/'treasury-yields.csv',index_col=0,parse_dates=True).apply(pd.to_numeric,errors='coerce').dropna()
    train,test=df.iloc[:-12],df.iloc[-12:]
    scaler=StandardScaler().fit(train);pca=PCA(n_components=3).fit(scaler.transform(train))
    factors=pca.transform(scaler.transform(train));var=VAR(factors).fit(maxlags=min(6,len(train)//10),ic='aic')
    lag=var.k_ar
    if lag:
        f=var.forecast(factors[-lag:],steps=len(test))
    else:
        f=np.tile(var.intercept,(len(test),1))
    pred=scaler.inverse_transform(pca.inverse_transform(f))
    baseline=np.tile(train.iloc[-1].to_numpy(),(len(test),1))
    pd.DataFrame(pred,index=test.index,columns=test.columns).to_csv(out/'yield-forecast.csv')
    plt.figure(figsize=(8,4));plt.plot(test.iloc[-1].to_numpy(),label='Held-out actual');plt.plot(pred[-1],label='PCA + VAR');plt.xticks(range(len(test.columns)),test.columns,rotation=30);plt.ylabel('Yield');plt.legend();plt.tight_layout();plt.savefig(out/'yield-curve.png',dpi=160)
    return finish(name,{'train_rows':len(train),'test_rows':len(test),'components':3,'var_lags':lag,'train_variance_explained':pca.explained_variance_ratio_.sum(),
      'test_rmse':mean_squared_error(test,pred)**.5,'persistence_rmse':mean_squared_error(test,baseline)**.5,
      'method_note':'PCA is fitted on training data only; forecast component signs are not arbitrarily flipped.'})

def cot_forecast():
    name='aps1052-cot-forecast';out=output(name)
    df=pd.read_csv(DATA/'cot-fx.csv',parse_dates=['Date']).sort_values('Date').set_index('Date')
    # Saved CAD=X series is USD/CAD (CAD per USD), despite the historical CADUSD column name.
    close=pd.to_numeric(df.CADUSD,errors='coerce');features=df.drop(columns=['CADUSD','log_CADUSD'],errors='ignore').apply(pd.to_numeric,errors='coerce')
    # Lag all external inputs by one weekly row to reduce COT publication-timing leakage.
    X=features.shift(1);X['log_fx']=np.log(close);X['weekly_return']=np.log(close).diff()
    target=np.log(close.shift(-1));all_data=X.assign(target=target).dropna();X=all_data.drop(columns='target');y=all_data.target
    split=int(len(X)*.8);a,b=X.iloc[:split],X.iloc[split:];ya,yb=y.iloc[:split],y.iloc[split:]
    ridge=GridSearchCV(make_pipeline(StandardScaler(),Ridge()),{'ridge__alpha':[.1,1.,10.,100.]},cv=TimeSeriesSplit(4,gap=1),scoring='neg_mean_squared_error').fit(a,ya)
    rf=RandomForestRegressor(n_estimators=100,max_depth=4,min_samples_leaf=5,random_state=SEED,n_jobs=2).fit(a,ya)
    preds={'ridge':ridge.predict(b),'random_forest':rf.predict(b),'persistence':b.log_fx.to_numpy()}
    metrics={k:{'rmse_log_fx':mean_squared_error(yb,p)**.5,'mae_log_fx':mean_absolute_error(yb,p)} for k,p in preds.items()}
    pd.DataFrame({'actual_log_fx':yb,**preds}).to_csv(out/'held-out-predictions.csv')
    return finish(name,{'train_weeks':len(a),'test_weeks':len(b),'features':X.shape[1],'models':metrics,
      'scope':'CPU baselines using the saved group-project dataset. This does not rerun the original neural-network/ARIMAX model suite. Exact release timestamps still require point-in-time validation.'})

def trading_workflow():
    name='aps1052-trading-workflow';out=output(name)
    df=pd.read_csv(DATA/'spy.csv',parse_dates=['Date']).sort_values('Date').set_index('Date')
    price=df['Adj Close'];ret=price.pct_change().fillna(0)
    fast=price.rolling(20,min_periods=20).mean();slow=price.rolling(80,min_periods=80).mean()
    signal=(fast>slow).astype(float);position=signal.shift(1).fillna(0)
    turnover=position.diff().abs().fillna(position.abs());strategy=position*ret-turnover*.001
    curves=pd.DataFrame({'buy_hold':(1+ret).cumprod(),'moving_average_after_costs':(1+strategy).cumprod()})
    curves.to_csv(out/'equity-curves.csv');curves.plot(figsize=(9,4));plt.ylabel('Growth of one unit');plt.tight_layout();plt.savefig(out/'equity-curves.png',dpi=160)
    return finish(name,{'rows':len(df),'start':str(df.index.min().date()),'end':str(df.index.max().date()),'turnover_cost_basis_points':10,
      'strategy_total_return':curves.iloc[-1,1]-1,'buy_hold_total_return':curves.iloc[-1,0]-1,
      'scope':'Educational fixed-rule backtest. Signal is lagged one day; no tax, financing, market impact or execution guarantees.'})

PROJECTS={'aps1052-seasonal-sales':seasonal_sales,'aps1052-yield-curve':yield_curve,'aps1052-cot-forecast':cot_forecast,'aps1052-trading-workflow':trading_workflow}
