"""MIE1624: statistical inference, classification and clustering.

Portable adaptations of the completed coursework, with methodological fixes
documented in the course README. These are not byte-for-byte notebook replays.
"""
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.oneway import anova_oneway
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from .common import DATA, SEED, plt, output, finish

def salary_statistics():
    name = 'mie1624-salary-statistics'; out = output(name)
    df = pd.read_csv(DATA / 'salary-2024.csv').dropna(subset=['ConvertedCompYearly'])
    before = len(df)
    q1, q3 = df.ConvertedCompYearly.quantile([.25, .75]); iqr = q3-q1
    df = df[df.ConvertedCompYearly.between(q1-1.5*iqr, q3+1.5*iqr)]
    a = df.loc[df.RemoteWork.eq('Remote'), 'ConvertedCompYearly'].to_numpy()
    b = df.loc[df.RemoteWork.eq('In-person'), 'ConvertedCompYearly'].to_numpy()
    assert min(len(a), len(b)) > 20
    test = stats.ttest_ind(a, b, equal_var=False)
    # Bootstrap the mean difference; do not treat bootstrap replicates as new observations.
    rng = np.random.default_rng(SEED)
    boot = np.array([rng.choice(a,len(a),replace=True).mean()-rng.choice(b,len(b),replace=True).mean() for _ in range(10000)])
    degrees = ["Bachelor", "Master", "Professional"]
    groups = [df.loc[df.EdLevel.fillna('').str.contains(d,case=False), 'ConvertedCompYearly'].to_numpy() for d in degrees]
    groups = [g for g in groups if len(g)>1]
    welch = anova_oneway(groups, use_var='unequal')
    df.groupby('RemoteWork').ConvertedCompYearly.agg(['size','mean','median','std']).to_csv(out/'group-summary.csv')
    plt.figure(figsize=(8,4)); plt.hist(boot,bins=60,color='#4B6587')
    plt.xlabel('Remote minus in-person mean salary (USD)');plt.ylabel('Bootstrap replicates')
    plt.tight_layout();plt.savefig(out/'bootstrap-mean-difference.png',dpi=160)
    return finish(name, {'input_rows':before,'retained_rows':len(df),'remote_n':len(a),'in_person_n':len(b),
       'mean_difference_usd':float(a.mean()-b.mean()),'bootstrap_95_percent_ci':np.quantile(boot,[.025,.975]),
       'welch_t':float(test.statistic),'welch_p':float(test.pvalue),'welch_anova_p':float(welch.pvalue),
       'bootstrap_resamples':10000,'interpretation':'Observational association; not a randomized A/B test or causal estimate.'})

def salary_classification():
    name='mie1624-salary-classification';out=output(name)
    df=pd.read_csv(DATA/'salary-2022.csv',dtype=str).dropna(subset=['Q29_buckets'])
    features=[c for c in df.columns if c!='Q29_buckets']
    counts=df.Q29_buckets.value_counts();df=df[df.Q29_buckets.isin(counts[counts>=10].index)]
    X_train,X_test,y_train,y_test=train_test_split(df[features],df.Q29_buckets,test_size=.25,stratify=df.Q29_buckets,random_state=SEED)
    preprocess=make_pipeline(SimpleImputer(strategy='most_frequent'),OneHotEncoder(handle_unknown='ignore'))
    model=make_pipeline(preprocess,LogisticRegression(max_iter=1500,C=.5,class_weight='balanced',random_state=SEED))
    scores=cross_val_score(model,X_train,y_train,cv=StratifiedKFold(3,shuffle=True,random_state=SEED),scoring='f1_macro')
    model.fit(X_train,y_train);pred=model.predict(X_test)
    baseline=DummyClassifier(strategy='most_frequent').fit(X_train,y_train).predict(X_test)
    pd.DataFrame(confusion_matrix(y_test,pred,labels=model.classes_),index=model.classes_,columns=model.classes_).to_csv(out/'confusion-matrix.csv')
    return finish(name,{'train_rows':len(X_train),'test_rows':len(X_test),'classes':len(model.classes_),
      'cv_macro_f1':scores.tolist(),'test_macro_f1':f1_score(y_test,pred,average='macro'),
      'test_balanced_accuracy':balanced_accuracy_score(y_test,pred),'baseline_macro_f1':f1_score(y_test,baseline,average='macro'),
      'scope':'Compact categorical-feature baseline; original notebook has additional feature selection and tuning.'})

def skill_clustering():
    name='mie1624-skill-clustering';out=output(name)
    df=pd.read_csv(DATA/'skill-features.csv');X=df.drop(columns='Skill').select_dtypes('number')
    X=X.copy();X['frequency']=(X.frequency-X.frequency.min())/(X.frequency.max()-X.frequency.min())
    model=KMeans(n_clusters=8,n_init=20,random_state=SEED).fit(X)
    df['kmeans_cluster']=model.labels_
    df['hierarchical_cluster']=AgglomerativeClustering(n_clusters=8,linkage='ward').fit_predict(X)
    df.to_csv(out/'skill-clusters.csv',index=False)
    xy=PCA(n_components=2).fit_transform(X)
    plt.figure(figsize=(8,5));plt.scatter(xy[:,0],xy[:,1],c=model.labels_,cmap='tab10');plt.xlabel('PC1');plt.ylabel('PC2')
    plt.tight_layout();plt.savefig(out/'skill-clusters.png',dpi=160)
    return finish(name,{'skills':len(df),'features':X.shape[1],'clusters':8,'silhouette':silhouette_score(X,model.labels_),
       'scope':'Clustering of the saved feature table from the original submission; web scraping and NER are not rerun.'})

PROJECTS={'mie1624-salary-statistics':salary_statistics,'mie1624-salary-classification':salary_classification,'mie1624-skill-clustering':skill_clustering}
