"""APS1070: portable versions of four completed modeling projects."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from .common import DATA, SEED, output, finish, plt

def basic_models():
    name='aps1070-basic-models'
    df=pd.read_csv(DATA/'bodyfat.csv');target='target'
    X_train,X_test,y_train,y_test=train_test_split(df.drop(columns=target),df[target],test_size=.25,random_state=SEED)
    threshold=float(y_train.median()); y_train=(y_train>threshold).astype(int);y_test=(y_test>threshold).astype(int)
    # Density directly determines body-fat estimates and is excluded to reduce target leakage.
    X_train=X_train.drop(columns=[c for c in X_train if c.lower()=='density']);X_test=X_test[X_train.columns]
    cv=StratifiedKFold(5,shuffle=True,random_state=SEED);results={}
    for label,base,grid in [('knn',make_pipeline(StandardScaler(),KNeighborsClassifier()),{'kneighborsclassifier__n_neighbors':[3,5,9,15]}),
      ('tree',DecisionTreeClassifier(random_state=SEED),{'max_depth':[2,3,4,6],'min_samples_leaf':[3,5]})]:
        model=GridSearchCV(base,grid,cv=cv,scoring='f1').fit(X_train,y_train)
        pred=model.predict(X_test)
        results[label]={'best_params':model.best_params_,'cv_f1':model.best_score_,'test_accuracy':accuracy_score(y_test,pred),'test_f1':f1_score(y_test,pred)}
    return finish(name,{'train_rows':len(X_train),'test_rows':len(X_test),'train_median_threshold':threshold,'models':results})

def satellite_anomalies():
    name='aps1070-satellite-anomalies';out=output(name)
    df=pd.read_csv(DATA/'landsat.csv');label=df.columns[-1]
    X=df.drop(columns=label);y=df[label].astype(int)
    assert set(y.unique())=={0,1}, 'Expected binary isanomaly labels in this course snapshot'
    # Learn normal behavior without anomalous examples; choose hyperparameters on validation only.
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,stratify=y,random_state=SEED)
    X_fit,X_val,y_fit,y_val=train_test_split(X_train,y_train,test_size=.25,stratify=y_train,random_state=SEED)
    scaler=StandardScaler().fit(X_fit[y_fit==0]);train=scaler.transform(X_fit[y_fit==0]);val=scaler.transform(X_val)
    candidates=[]
    for n in [1,2,4]:
        g=GaussianMixture(n_components=n,covariance_type='diag',n_init=2,random_state=SEED).fit(train)
        candidates.append((roc_auc_score(y_val,-g.score_samples(val)),g))
    va,model=max(candidates,key=lambda z:z[0]);scores=-model.score_samples(scaler.transform(X_test))
    pd.DataFrame({'is_anomaly':y_test,'anomaly_score':scores}).to_csv(out/'held-out-scores.csv',index=False)
    return finish(name,{'train_rows':len(X_fit),'validation_rows':len(X_val),'test_rows':len(X_test),'anomalous_label':1,
       'chosen_components':model.n_components,'validation_auc':va,'test_auc':roc_auc_score(y_test,scores),
       'test_anomalies':int(y_test.sum()),'test_normal':int((y_test==0).sum()),
       'interpretation':'Small anomaly count limits precision of the held-out AUC estimate.'})

def sales_pca():
    name='aps1070-sales-pca';out=output(name)
    df=pd.read_csv(DATA/'sales-by-family.csv')
    # Input rows are product families and columns are dates. Model days as observations.
    X=df.set_index('family').T.apply(pd.to_numeric).dropna()
    X.index=pd.to_datetime(X.index);X=X.sort_index()
    assert X.shape[1]>=3
    split=int(len(X)*.8);train,test=X.iloc[:split],X.iloc[split:]
    scaler=StandardScaler().fit(train);a=scaler.transform(train);b=scaler.transform(test)
    rows=[]
    for k in sorted(set([1,2,3,min(5,X.shape[1]),min(10,X.shape[1]),X.shape[1]])):
        pca=PCA(n_components=k,random_state=SEED).fit(a);rebuilt=pca.inverse_transform(pca.transform(b))
        rows.append({'components':k,'train_explained_variance':pca.explained_variance_ratio_.sum(),'test_standardized_mse':mean_squared_error(b,rebuilt)})
    pd.DataFrame(rows).to_csv(out/'reconstruction-errors.csv',index=False)
    plt.figure(figsize=(7,4));plt.plot([r['components'] for r in rows],[r['test_standardized_mse'] for r in rows],marker='o')
    plt.xlabel('Principal components');plt.ylabel('Held-out reconstruction MSE');plt.tight_layout();plt.savefig(out/'reconstruction.png',dpi=160)
    return finish(name,{'train_rows':len(train),'test_rows':len(test),'features':X.shape[1],'reconstruction':rows})

def aileron_regression():
    name='aps1070-aileron-regression';out=output(name)
    df=pd.read_csv(DATA/'ailerons.csv');X=df.drop(columns='goal');y=df.goal
    a,b,ya,yb=train_test_split(X,y,test_size=.2,random_state=SEED)
    scaler=StandardScaler().fit(a);a=np.c_[np.ones(len(a)),scaler.transform(a)];b=np.c_[np.ones(len(b)),scaler.transform(b)]
    # Stable least squares avoids explicitly inverting an ill-conditioned X'X matrix.
    exact=np.linalg.lstsq(a,ya,rcond=None)[0]
    beta=np.zeros(a.shape[1]);lr=.9/(np.linalg.norm(a,2)**2/len(a));loss=[]
    for i in range(2500):
        beta-=lr*(a.T@(a@beta-ya.to_numpy()))/len(a)
        if i%100==0:loss.append({'iteration':i,'train_mse':float(np.mean((a@beta-ya)**2))})
    pd.DataFrame(loss).to_csv(out/'gradient-descent-loss.csv',index=False)
    return finish(name,{'train_rows':len(a),'test_rows':len(b),'least_squares_test_rmse':mean_squared_error(yb,b@exact)**.5,
      'gradient_descent_test_rmse':mean_squared_error(yb,b@beta)**.5,'iterations':2500,'learning_rate':lr})

PROJECTS={'aps1070-basic-models':basic_models,'aps1070-satellite-anomalies':satellite_anomalies,'aps1070-sales-pca':sales_pca,'aps1070-aileron-regression':aileron_regression}
