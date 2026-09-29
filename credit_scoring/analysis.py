"""Reproducible credit scoring analysis using the team's cleaned CSV."""
from pathlib import Path
import os,json,time,platform,warnings,copy
ROOT=Path(__file__).resolve().parent.parent
os.environ.setdefault('OMP_NUM_THREADS','4')
import numpy as np,pandas as pd
import scipy
from scipy.special import expit,logit
from scipy.stats import norm,chi2,spearmanr
import sklearn,xgboost,shap,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator,TransformerMixin,clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler,SplineTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedGroupKFold,StratifiedKFold,train_test_split,GridSearchCV
from sklearn.metrics import roc_auc_score,average_precision_score,log_loss,brier_score_loss,roc_curve,precision_recall_curve
from sklearn.calibration import calibration_curve
from joblib import dump
from concurrent.futures import ThreadPoolExecutor

SEED=20260927
TARGET='default payment next month'
SOURCE=ROOT/'data'/'cleaned'/'Du_lieu_sach_Chu_de_1.csv'
SOURCE_TO_MODEL={
    'customer_id':'ID',
    'credit_limit_ntd':'LIMIT_BAL',
    'sex':'SEX',
    'education':'EDUCATION',
    'marital_status':'MARRIAGE',
    'age':'AGE',
    'repayment_status_sep':'PAY_0',
    'repayment_status_aug':'PAY_2',
    'repayment_status_jul':'PAY_3',
    'repayment_status_jun':'PAY_4',
    'repayment_status_may':'PAY_5',
    'repayment_status_apr':'PAY_6',
    'bill_amount_sep_ntd':'BILL_AMT1',
    'bill_amount_aug_ntd':'BILL_AMT2',
    'bill_amount_jul_ntd':'BILL_AMT3',
    'bill_amount_jun_ntd':'BILL_AMT4',
    'bill_amount_may_ntd':'BILL_AMT5',
    'bill_amount_apr_ntd':'BILL_AMT6',
    'payment_amount_sep_ntd':'PAY_AMT1',
    'payment_amount_aug_ntd':'PAY_AMT2',
    'payment_amount_jul_ntd':'PAY_AMT3',
    'payment_amount_jun_ntd':'PAY_AMT4',
    'payment_amount_may_ntd':'PAY_AMT5',
    'payment_amount_apr_ntd':'PAY_AMT6',
    'default_next_month':TARGET,
}
MODEL_COLUMNS=list(SOURCE_TO_MODEL.values())
CAT=['EDUCATION','MARRIAGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']
MODELS=['Logistic_WoE','Random_Forest','XGBoost']
OUT=ROOT/'results';FIG=ROOT/'figures';DATA=ROOT/'data'/'generated';ART=ROOT/'models'
for d in [OUT,FIG,DATA,ART]:d.mkdir(exist_ok=True,parents=True)
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'figure.dpi':150,'savefig.dpi':300,'axes.spines.top':False,'axes.spines.right':False})
LOG=[]
def note(x):
    line=f'{time.strftime("%H:%M:%S")} {x}';print(line,flush=True);LOG.append(line)
    (OUT/'run_log.txt').write_text('\n'.join(LOG),encoding='utf-8')
def save(df,name):df.to_csv(OUT/name,index=False,encoding='utf-8-sig')
def load_data():
    if not SOURCE.is_file():raise FileNotFoundError(f'Clean input not found: {SOURCE}')
    df=pd.read_csv(SOURCE)
    if list(df.columns)!=list(SOURCE_TO_MODEL):raise ValueError('Clean CSV columns do not match the expected data dictionary')
    if len(df)!=30000 or df.isna().any().any():raise ValueError('Expected 30,000 complete records')
    if any(not pd.api.types.is_integer_dtype(df[c]) for c in df):raise ValueError('All source fields must be integers')
    df=df.rename(columns=SOURCE_TO_MODEL)[MODEL_COLUMNS].sort_values('ID',kind='stable').reset_index(drop=True)
    if not df.ID.is_unique or not df.ID.between(1,30000).all():raise ValueError('Invalid or duplicate customer IDs')
    if not set(df[TARGET].unique()).issubset({0,1}):raise ValueError('The target must be binary')
    return df
def signatures(df):return pd.util.hash_pandas_object(df,index=False).to_numpy().astype(str)

class WoETransformer(TransformerMixin,BaseEstimator):
    def __init__(self,n_bins=10,iv_min=.02,smoothing=.5):self.n_bins=n_bins;self.iv_min=iv_min;self.smoothing=smoothing
    def fit(self,X,y):
        X=pd.DataFrame(X);y=np.asarray(y);self.columns_=list(X.columns);self.spec_={};self.iv_={};rows=[]
        for c in self.columns_:
            v=X[c].to_numpy();edges=None
            if c in CAT:
                cats=np.sort(np.unique(v));bins=v
            else:
                edges=np.unique(np.quantile(v,np.linspace(0,1,self.n_bins+1)))[1:-1];bins=np.digitize(v,edges);cats=np.sort(np.unique(bins))
            k=len(cats);ng=(y==0).sum();nb=(y==1).sum();mapping={};iv=0
            for b in cats:
                keep=bins==b;g=int(((y==0)&keep).sum());bad=int(((y==1)&keep).sum())
                G=(g+self.smoothing)/(ng+self.smoothing*k);B=(bad+self.smoothing)/(nb+self.smoothing*k)
                w=float(np.log(G/B));mapping[float(b)]=w;piece=(G-B)*w;iv+=piece
                if edges is None:label=str(b)
                else:
                    ix=int(b);left=float(edges[ix-1]) if ix else None;right=float(edges[ix]) if ix<len(edges) else None
                    label=f'[{left},{right})'
                rows.append({'variable':c,'bin':str(b),'definition':label,'good':g,'bad':bad,'woe':w,'iv_component':piece})
            self.spec_[c]=(edges,mapping);self.iv_[c]=iv
        self.selected_=[c for c in self.columns_ if self.iv_[c]>=self.iv_min]
        if not self.selected_:self.selected_=[max(self.iv_,key=self.iv_.get)]
        self.rows_=rows;return self
    def transform(self,X):
        X=pd.DataFrame(X,columns=self.columns_);result=[]
        for c in self.selected_:
            edges,mapping=self.spec_[c];v=X[c].to_numpy();bins=v if edges is None else np.digitize(v,edges)
            result.append(np.array([mapping.get(float(b),0.) for b in bins]))
        return np.column_stack(result)
    def get_feature_names_out(self,input_features=None):return np.array(self.selected_,dtype=object)

def preprocess(cols):
    cats=[c for c in CAT if c in cols];nums=[c for c in cols if c not in cats]
    return ColumnTransformer([('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cats),('num','passthrough',nums)])
def make_model(name,cols,params=None,weighted=False):
    if name=='Logistic_WoE':
        m=Pipeline([('prep',WoETransformer()),('clf',LogisticRegression(C=1,max_iter=2000,class_weight='balanced' if weighted else None))])
    elif name=='Random_Forest':
        m=Pipeline([('prep',preprocess(cols)),('clf',RandomForestClassifier(n_estimators=160,max_depth=10,min_samples_leaf=10,max_features='sqrt',random_state=SEED,n_jobs=4,class_weight='balanced' if weighted else None))])
    elif name=='XGBoost':
        m=Pipeline([('prep',preprocess(cols)),('clf',xgboost.XGBClassifier(n_estimators=200,max_depth=3,learning_rate=.05,subsample=.9,colsample_bytree=.9,min_child_weight=10,reg_lambda=5,tree_method='hist',n_jobs=4,random_state=SEED,eval_metric='logloss',scale_pos_weight=3.52 if weighted else 1))])
    elif name=='Logistic_Spline':
        cats=[c for c in CAT if c in cols];nums=[c for c in cols if c not in cats]
        pre=ColumnTransformer([('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cats),('num',Pipeline([('scale',StandardScaler()),('spline',SplineTransformer(n_knots=4,degree=3,include_bias=False))]),nums)])
        m=Pipeline([('prep',pre),('clf',LogisticRegression(C=.1,max_iter=2000))])
    if params:m.set_params(**params)
    return m
GRIDS={
'Logistic_WoE':{'prep__n_bins':[5,10],'clf__C':[.1,1]},
'Random_Forest':{'clf__max_depth':[6,10],'clf__min_samples_leaf':[10,30]},
'XGBoost':{'clf__max_depth':[3,5],'clf__learning_rate':[.05,.1]}}

class Calibrator:
    def __init__(self,method):self.method=method
    def fit(self,p,y):
        if self.method=='sigmoid':self.model=LogisticRegression(C=1e6,max_iter=2000).fit(logit(np.clip(p,1e-6,1-1e-6)).reshape(-1,1),y)
        elif self.method=='isotonic':self.model=IsotonicRegression(out_of_bounds='clip').fit(p,y)
        return self
    def predict(self,p):
        if self.method=='none':return np.asarray(p)
        if self.method=='sigmoid':return self.model.predict_proba(logit(np.clip(p,1e-6,1-1e-6)).reshape(-1,1))[:,1]
        return self.model.predict(p)
def choose_calibration(oof,y,groups,name):
    cv=StratifiedGroupKFold(5,shuffle=True,random_state=SEED+55);rows=[]
    for method in ['none','sigmoid','isotonic']:
        pp=np.empty(len(y))
        for tr,va in cv.split(oof,y,groups):pp[va]=Calibrator(method).fit(oof[tr],y[tr]).predict(oof[va])
        rows.append({'model':name,'calibration':method,'development_cv_log_loss':log_loss(y,np.clip(pp,1e-6,1-1e-6)),'development_cv_brier':brier_score_loss(y,pp)})
    best=min(rows,key=lambda r:r['development_cv_log_loss'])['calibration']
    return Calibrator(best).fit(oof,y),rows
def metrics(y,p):
    p=np.clip(np.asarray(p),1e-6,1-1e-6);fpr,tpr,_=roc_curve(y,p)
    auc=roc_auc_score(y,p)
    return {'auc':auc,'gini':2*auc-1,'ks':float(np.max(tpr-fpr)),'average_precision':average_precision_score(y,p),'log_loss':log_loss(y,p),'brier':brier_score_loss(y,p)}
def utility(y,p,ratio=5):return np.where(p<=1/(1+ratio),np.where(y==0,1.,-float(ratio)),0.)
def calibration_stats(y,p):
    z=logit(np.clip(p,1e-6,1-1e-6));cal=LogisticRegression(C=1e6,max_iter=2000).fit(z.reshape(-1,1),y)
    return float(cal.intercept_[0]),float(cal.coef_[0,0])
def bootstrap_indices(groups,n=2000,seed=SEED+99):
    rng=np.random.default_rng(seed);unq=np.unique(groups);by={g:np.flatnonzero(groups==g) for g in unq}
    for _ in range(n):yield np.concatenate([by[g] for g in rng.choice(unq,len(unq),replace=True)])
def paired_ci(y,a,b,groups,n=1000):
    values=[]
    for ix in bootstrap_indices(groups,n):values.append(roc_auc_score(y[ix],a[ix])-roc_auc_score(y[ix],b[ix]))
    return np.quantile(values,[.025,.975])
def feature_map(model,cols):
    if model=='Logistic_WoE':return cols
    pre=model.named_steps['prep'];mapping=[]
    encoder=pre.named_transformers_['cat']
    for col,cats in zip(pre.transformers_[0][2],encoder.categories_):mapping.extend([col]*len(cats))
    mapping.extend(pre.transformers_[1][2]);return mapping
def explain(model,name,X,background=None,path_dependent=False):
    z=model.named_steps['prep'].transform(X);clf=model.named_steps['clf']
    if name=='Logistic_WoE':
        base=float(clf.intercept_[0]);sv=z*clf.coef_[0];mapping=model.named_steps['prep'].selected_;raw=clf.decision_function(z);scale='log_odds'
    else:
        if path_dependent:e=shap.TreeExplainer(clf,feature_perturbation='tree_path_dependent',model_output='raw')
        else:
            bg=model.named_steps['prep'].transform(background)
            e=shap.TreeExplainer(clf,data=bg,feature_perturbation='interventional',model_output='raw')
        # XGBoost 3.4 enables categorical capability globally even for numeric trees.
        # Reset only the capability flag after verifying no categorical feature or split.
        if name=='XGBoost' and not path_dependent:
            assert clf.get_booster().feature_types is None
            trees=[json.loads(t) for t in clf.get_booster().get_dump(dump_format='json')]
            def numeric_tree(t):
                return not isinstance(t.get('split_condition'),list) and all(numeric_tree(c) for c in t.get('children',[]))
            assert all(numeric_tree(t) for t in trees) and e.model.cat_feature_indices is None
            e.model._xgb_enable_categorical=False
        sv=e.shap_values(z,check_additivity=False);base=e.expected_value
        if isinstance(sv,list):sv=sv[1]
        if np.asarray(sv).ndim==3:sv=np.asarray(sv)[:,:,1];base=np.asarray(base)[1]
        if np.asarray(base).ndim:base=np.asarray(base).ravel()[-1]
        if name=='XGBoost':raw=clf.predict(z,output_margin=True);scale='log_odds'
        else:raw=clf.predict_proba(z)[:,1];scale='probability'
        mapping=feature_map(model,list(X.columns))
    sv=np.asarray(sv);grouped=np.zeros((len(X),len(X.columns)))
    for j,c in enumerate(mapping):grouped[:,list(X.columns).index(c)]+=sv[:,j]
    return grouped,float(base),np.asarray(raw),scale
def group_family(col):
    if col.startswith('BILL_AMT'):return 'BILL'
    if col.startswith('PAY_AMT'):return 'PAYMENT'
    if col.startswith('PAY_'):return 'DELINQUENCY'
    return col
def family_importance(sv,cols):
    families=sorted(set(map(group_family,cols)))
    val=np.column_stack([sv[:,[i for i,c in enumerate(cols) if group_family(c)==f]].sum(axis=1) for f in families])
    return families,np.abs(val).mean(axis=0)

def run():
    started=time.time();df=load_data();y=df[TARGET].to_numpy().astype(int);X=df.drop(columns=['ID','SEX',TARGET]);cols=list(X.columns)
    groups=signatures(df.drop(columns=['ID',TARGET]));split=StratifiedGroupKFold(5,shuffle=True,random_state=SEED)
    dev,test=next(split.split(X,y,groups));Xd=X.iloc[dev].reset_index(drop=True);yd=y[dev];gd=groups[dev];Xt=X.iloc[test].reset_index(drop=True);yt=y[test];gt=groups[test]
    assert not set(gd)&set(gt)
    df['split']=np.where(np.isin(np.arange(len(df)),test),'test','development');df[['ID','split']].to_csv(DATA/'split_membership.csv',index=False,encoding='utf-8-sig')
    save(pd.DataFrame([{'partition':'full','n':len(df),'defaults':sum(y),'default_rate':np.mean(y)},{'partition':'development','n':len(dev),'defaults':sum(yd),'default_rate':np.mean(yd)},{'partition':'test','n':len(test),'defaults':sum(yt),'default_rate':np.mean(yt)}]),'table_02_sample_flow.csv')
    dsc=df.drop(columns='split').describe().T.reset_index().rename(columns={'index':'variable'});save(dsc,'table_01_descriptive.csv')
    versions={k:v.__version__ for k,v in [('numpy',np),('pandas',pd),('scipy',scipy),('scikit-learn',sklearn),('xgboost',xgboost),('shap',shap),('matplotlib',matplotlib)]}
    versions['python']=platform.python_version();(OUT/'software_versions.json').write_text(json.dumps(versions,indent=2),encoding='utf-8')
    note(f'Data loaded: development {len(dev)}, test {len(test)}; overlapping predictor groups 0')
    fitted={};calibrators={};oofs={};params={};cvrows=[];calrows=[];tunerows=[]
    for name in MODELS:
        note(f'Nested development validation {name}')
        outer=StratifiedGroupKFold(5,shuffle=True,random_state=SEED+1);oof=np.empty(len(dev))
        for fold,(tr,va) in enumerate(outer.split(Xd,yd,gd),1):
            inner=StratifiedGroupKFold(3,shuffle=True,random_state=SEED+fold+10)
            search=GridSearchCV(make_model(name,cols),GRIDS[name],cv=inner,scoring='roc_auc',refit=True,n_jobs=1,error_score='raise').fit(Xd.iloc[tr],yd[tr],groups=gd[tr])
            oof[va]=search.predict_proba(Xd.iloc[va])[:,1]
            cvrows.append({'model':name,'outer_fold':fold,**metrics(yd[va],oof[va]),'best_params':json.dumps(search.best_params_)})
            note(f'{name} outer fold {fold} complete')
        finalcv=StratifiedGroupKFold(5,shuffle=True,random_state=SEED+20)
        final=GridSearchCV(make_model(name,cols),GRIDS[name],cv=finalcv,scoring='roc_auc',refit=True,n_jobs=1,error_score='raise').fit(Xd,yd,groups=gd)
        fitted[name]=final.best_estimator_;params[name]=final.best_params_;oofs[name]=oof
        calibrators[name],rows=choose_calibration(oof,yd,gd,name);calrows+=rows
        for par,score in zip(final.cv_results_['params'],final.cv_results_['mean_test_score']):tunerows.append({'model':name,'params':json.dumps(par),'development_cv_auc':score})
        dump({'model':fitted[name],'calibrator':calibrators[name],'params':params[name]},ART/f'{name}.joblib')
    save(pd.DataFrame(cvrows),'development_nested_cv.csv');save(pd.DataFrame(calrows),'calibration_selection.csv');save(pd.DataFrame(tunerows),'hyperparameter_search.csv')
    (OUT/'selected_parameters.json').write_text(json.dumps(params,indent=2),encoding='utf-8')
    oofdf=pd.DataFrame({'ID':df.iloc[dev]['ID'].to_numpy(),'y':yd,'group':gd})
    for name in MODELS:oofdf[name]=oofs[name]
    oofdf.to_csv(DATA/'development_oof_predictions.csv',index=False,encoding='utf-8-sig')
    note('Development choices complete; opening fixed test once for final evaluation')
    pred=pd.DataFrame({'ID':df.iloc[test]['ID'].to_numpy(),'y':yt,'SEX':df.iloc[test]['SEX'].to_numpy(),'AGE':df.iloc[test]['AGE'].to_numpy(),'group':gt})
    allprobs={};metricrows=[]
    for name in MODELS:
        raw=fitted[name].predict_proba(Xt)[:,1]
        for method in ['none','sigmoid','isotonic']:
            cal=Calibrator(method).fit(oofs[name],yd);p=cal.predict(raw);key=name+'__'+method;allprobs[key]=p;pred[key]=p
            a,b=calibration_stats(yt,p);metricrows.append({'model':name,'calibration':method,'selected':method==calibrators[name].method,**metrics(yt,p),'calibration_intercept':a,'calibration_slope':b})
        pred[name+'__selected']=calibrators[name].predict(raw)
    pred['scorecard_points']=600+20/np.log(2)*(logit(1-np.clip(pred['Logistic_WoE__none'],1e-6,1-1e-6))-np.log(50))
    pred.to_csv(DATA/'test_predictions.csv',index=False,encoding='utf-8-sig');save(pd.DataFrame(metricrows),'table_03_predictive_performance.csv')
    woe=fitted['Logistic_WoE'].named_steps['prep'];save(pd.DataFrame(woe.rows_),'woe_bins.csv')
    save(pd.DataFrame([{'variable':c,'iv':v,'selected':c in woe.selected_} for c,v in woe.iv_.items()]),'information_value.csv')
    clf=fitted['Logistic_WoE'].named_steps['clf'];coef=dict(zip(woe.selected_,clf.coef_[0]));B=20/np.log(2)
    pts=pd.DataFrame(woe.rows_);pts['selected']=pts.variable.isin(woe.selected_);pts['coefficient']=pts.variable.map(coef);pts['point_contribution']=-B*pts.coefficient*pts.woe
    pts['base_points']=600-B*np.log(50)-B*float(clf.intercept_[0]);save(pts,'scorecard_points.csv')
    # Paired grouped uncertainty for the principal hypotheses.
    rawx=pred['XGBoost__none'].to_numpy();rawl=pred['Logistic_WoE__none'].to_numpy();px=pred['XGBoost__selected'].to_numpy()
    h1=roc_auc_score(yt,rawx)-roc_auc_score(yt,rawl);h2=float(np.mean(utility(yt,px)-utility(yt,rawx)))
    sx=pred.SEX.to_numpy();good=yt==0;reject=px>1/6
    h3=float(np.mean(reject[good&(sx==1)])-np.mean(reject[good&(sx==2)]))
    boot=[];mc=[]
    for j,ix in enumerate(bootstrap_indices(gt,2000),1):
        yy=yt[ix];a=rawx[ix];b=rawl[ix];pp=px[ix];ss=sx[ix];rr=pp>1/6;gg=yy==0
        boot.append([roc_auc_score(yy,a)-roc_auc_score(yy,b),np.mean(utility(yy,pp)-utility(yy,a)),np.mean(rr[gg&(ss==1)])-np.mean(rr[gg&(ss==2)])])
        for name in MODELS:
            for method in ['none',calibrators[name].method]:
                p=pred[name+'__'+method].to_numpy()[ix];m=metrics(yy,p)
                mc.append({'replicate':j,'model':name,'calibration':method,**m})
    boot=np.asarray(boot);save(pd.DataFrame(boot,columns=['h1_auc_difference','h2_utility_difference','h3_fpr_difference']),'bootstrap_hypotheses.csv')
    cis=[]
    for (name,method),frame in pd.DataFrame(mc).groupby(['model','calibration']):
        for metric in ['auc','gini','ks','average_precision','log_loss','brier']:
            lo,hi=np.quantile(frame[metric],[.025,.975]);cis.append({'model':name,'calibration':method,'metric':metric,'lower_95':lo,'upper_95':hi})
    save(pd.DataFrame(cis),'predictive_confidence_intervals.csv')
    estimates=[h1,h2,h3];hp=[]
    for k,v in enumerate(estimates):
        lo,hi=np.quantile(boot[:,k],[.025,.975]);pv=(1+np.sum(np.abs(boot[:,k]-v)>=abs(v)))/2001
        hp.append({'hypothesis':f'H{k+1}','status':'confirmatory' if k<2 else 'exploratory','estimate':v,'lower_95':lo,'upper_95':hi,'p_two_sided_centered_bootstrap':float(pv)})
    order=np.argsort([r['p_two_sided_centered_bootstrap'] for r in hp[:2]]);previous=0
    for rank,k in enumerate(order):
        previous=max(previous,min(1,(2-rank)*hp[k]['p_two_sided_centered_bootstrap']));hp[k]['p_holm']=previous
    for k,row in enumerate(hp):
        pvalue=row.get('p_holm',row['p_two_sided_centered_bootstrap']);row['supported']=bool(pvalue<.05 and (row['estimate']>0 if k<2 else True));row['scope']='fixed grouped test; conditional on fitted models'
    save(pd.DataFrame(hp),'table_08_hypothesis_summary.csv')
    note('Primary paired bootstrap complete')
    # Decision scenarios and full threshold curves.
    decisions=[];thresholds=[]
    for name in MODELS:
        for method in ['none',calibrators[name].method]:
            p=pred[name+'__'+method].to_numpy()
            for ratio in [2,5,10,20]:
                acc=p<=1/(1+ratio);decisions.append({'model':name,'calibration':method,'cost_ratio':ratio,'threshold':1/(1+ratio),'acceptance_rate':np.mean(acc),'default_rate_accepted':float(np.mean(yt[acc])) if acc.any() else np.nan,'utility_per_client':np.mean(utility(yt,p,ratio))})
            for tau in np.linspace(.01,.8,160):
                accepted=p<=tau;thresholds.append({'model':name,'calibration':method,'threshold':tau,'acceptance_rate':np.mean(accepted),'default_rate_accepted':float(np.mean(yt[accepted])) if accepted.any() else np.nan,'utility_ratio_5':float(np.mean(np.where(accepted,np.where(yt==0,1,-5),0)))})
    save(pd.DataFrame(decisions),'table_05_decision_scenarios.csv');save(pd.DataFrame(thresholds),'threshold_curves.csv')
    # Audit both sex and age, with explicit denominators.
    age=pd.cut(pred.AGE,bins=[20,29,44,59,np.inf],labels=['21 to 29','30 to 44','45 to 59','60 and above']).astype(str).to_numpy();fair=[]
    scopes={'sex':sx.astype(str),'age':age,'sex_by_age':np.char.add(np.char.add(sx.astype(str),' / '),age)}
    for name in MODELS:
        p=pred[name+'__selected'].to_numpy();r=p>1/6
        for dimension,labels in scopes.items():
            for label in sorted(set(labels)):
                sel=labels==label;g=sel&(yt==0);b=sel&(yt==1)
                vals={};
                for met,den,num in [('fpr',g,r),('tpr',b,r),('fnr',b,~r)]:
                    n=int(den.sum());v=float(np.mean(num[den])) if n else np.nan
                    if n:
                        z=1.95996398454;mid=(v+z*z/(2*n))/(1+z*z/n);half=z*np.sqrt(v*(1-v)/n+z*z/(4*n*n))/(1+z*z/n);lo,hi=mid-half,mid+half
                    else:lo=hi=np.nan
                    vals.update({met:v,met+'_lower_95_wilson':lo,met+'_upper_95_wilson':hi})
                fair.append({'model':name,'dimension':dimension,'group':label,'n':int(sel.sum()),'nondefaults':int(g.sum()),'defaults':int(b.sum()),'acceptance_rate':float(np.mean(~r[sel])),'group_brier':brier_score_loss(yt[sel],p[sel]),'mean_pd':np.mean(p[sel]),'observed_default_rate':np.mean(yt[sel]),**vals})
    save(pd.DataFrame(fair),'table_06_fairness.csv')
    # Exploratory pooled sex by age interactions among observed nondefaults.
    mask=yt==0;G=(sx[mask]==1).astype(float);a=age[mask];agelevels=['30 to 44','45 to 59','60 and above'];A=np.column_stack([(a==k).astype(float) for k in agelevels])
    Z=np.column_stack([np.ones(mask.sum()),G,A,G[:,None]*A]);d=reject[mask].astype(float);inv=np.linalg.pinv(Z.T@Z);beta=inv@Z.T@d;res=d-Z@beta;vc=inv@(Z.T@(Z*res[:,None]**2))@inv*len(d)/(len(d)-Z.shape[1]);se=np.sqrt(np.maximum(np.diag(vc),0));pv=2*norm.sf(np.abs(beta/np.maximum(se,1e-12)))
    labels=['intercept','male','age_30_44','age_45_59','age_60_plus','male_by_age_30_44','male_by_age_45_59','male_by_age_60_plus']
    save(pd.DataFrame({'term':labels,'coefficient':beta,'hc1_se':se,'p_two_sided':pv}),'fairness_interactions_exploratory.csv')
    ids=np.arange(5,8);wald=float(beta[ids]@np.linalg.pinv(vc[np.ix_(ids,ids)])@beta[ids]);(OUT/'fairness_interaction_joint_test.json').write_text(json.dumps({'wald_chi2':wald,'df':3,'p':float(chi2.sf(wald,3)),'scope':'exploratory statistical association, not causal'}),encoding='utf-8')
    # Fixed SHAP profiles: development near threshold, plus stratified held-out explanations.
    devcal=calibrators['XGBoost'].predict(oofs['XGBoost']);evaldev=np.argsort(np.abs(devcal-1/6))[:80];E=Xd.iloc[evaldev];rng=np.random.default_rng(SEED+70)
    bgix=rng.choice(len(dev),32,replace=False);BG=Xd.iloc[bgix]
    testeval=np.sort(np.concatenate([rng.choice(np.flatnonzero(yt==k),192,replace=False) for k in [0,1]]));Et=Xt.iloc[testeval]
    explanation_rows=[];importance=[];baseimports={}
    for name in MODELS:
        note(f'Explanations {name}')
        sv,base,raw,scale=explain(fitted[name],name,Et,BG);residual=np.abs(base+sv.sum(axis=1)-raw)
        for ii,row in enumerate(sv):
            for jj,col in enumerate(cols):explanation_rows.append({'ID':int(pred.iloc[testeval[ii]].ID),'model':name,'variable':col,'contribution':row[jj],'base_value':base,'output_scale':scale,'model_output':raw[ii]})
        for col,v in zip(cols,np.abs(sv).mean(axis=0)):importance.append({'model':name,'variable':col,'mean_absolute_contribution':v,'output_scale':scale})
        svd,bd,rd,scd=explain(fitted[name],name,E,BG);baseimports[name]=(np.abs(svd).mean(axis=0),family_importance(svd,cols)[1])
        save(pd.DataFrame({'model':[name],'evaluation_n':[len(Et)],'max_additivity_error':[residual.max()],'mean_additivity_error':[residual.mean()],'output_scale':[scale]}),f'additivity_{name}.csv')
        assert residual.max()<.01,(name,residual.max())
        if name=='XGBoost':
            imp=np.abs(sv).mean(axis=0);chosen=np.argsort(imp)[-12:]
            fig,ax=plt.subplots(figsize=(7,4.8));ax.barh(np.array(cols)[chosen],imp[chosen],color='#264764');ax.set_xlabel('Mean absolute SHAP contribution in log odds');fig.tight_layout();fig.savefig(FIG/'figure_02_shap_global.png');fig.savefig(FIG/'figure_02_shap_global.svg');plt.close(fig)
            sel=np.argsort(np.abs(devcal[evaldev]-1/6))[:2];local=[]
            fig,axes=plt.subplots(1,2,figsize=(10,4.7))
            for ax,j in zip(axes,sel):
                v=svd[j];top=np.argsort(np.abs(v))[-8:];ax.barh(np.array(cols)[top],v[top],color=np.where(v[top]>0,'#a54d42','#264764'));ax.set_title(f'Development ID {int(df.iloc[dev[evaldev[j]]].ID)}');ax.set_xlabel('SHAP contribution in log odds')
                for c,vv in zip(cols,v):local.append({'ID':int(df.iloc[dev[evaldev[j]]].ID),'variable':c,'contribution':vv,'base_log_odds':bd,'predicted_pd':float(expit(rd[j])),'selection':'closest to threshold using development OOF PD'})
            fig.tight_layout();fig.savefig(FIG/'figure_02_shap_local.png');fig.savefig(FIG/'figure_02_shap_local.svg');plt.close(fig);save(pd.DataFrame(local),'shap_local_development.csv')
    save(pd.DataFrame(explanation_rows),'shap_test_contributions.csv');save(pd.DataFrame(importance),'shap_global_importance.csv')
    # 200 complete grouped bootstrap refits, fixed hyperparameters and fixed development profiles.
    note('Starting 200 grouped bootstrap refits per model for explanation stability')
    bootix=list(bootstrap_indices(gd,200,SEED+88));stability=[]
    for name in MODELS:
        original,famorig=baseimports[name];toporiginal=set(np.argsort(original)[-5:])
        def one(item):
            k,ix=item;m=make_model(name,cols,params[name]);m.named_steps['clf'].set_params(n_jobs=2) if name!='Logistic_WoE' else None
            m.fit(Xd.iloc[ix],yd[ix]);sv,b,r,sc=explain(m,name,E,BG);imp=np.abs(sv).mean(axis=0);families,fimp=family_importance(sv,cols)
            top=set(np.argsort(imp)[-5:]);return {'model':name,'replicate':k+1,'rank_spearman_variables':spearmanr(original,imp).statistic,'rank_spearman_families':spearmanr(famorig,fimp).statistic,'top5_jaccard':len(top&toporiginal)/len(top|toporiginal),'max_additivity_error':float(np.max(np.abs(b+sv.sum(axis=1)-r))),'output_scale':sc}
        with ThreadPoolExecutor(max_workers=2) as ex:
            for row in ex.map(one,enumerate(bootix)):
                stability.append(row)
                if row['replicate']%25==0:note(f'{name} explanation bootstrap {row["replicate"]}/200')
        save(pd.DataFrame(stability),'explanation_stability_bootstrap.csv')
    ss=pd.DataFrame(stability);summ=[]
    for name,frame in ss.groupby('model'):
        for met in ['rank_spearman_variables','rank_spearman_families','top5_jaccard']:
            lo,hi=np.quantile(frame[met],[.025,.975]);summ.append({'model':name,'metric':met,'mean':frame[met].mean(),'median':frame[met].median(),'lower_95_refit_distribution':lo,'upper_95_refit_distribution':hi,'replicates':len(frame)})
    save(pd.DataFrame(summ),'table_04_explanation_stability.csv')
    # Robustness checks map exactly to R1 through R9.
    robust=[]
    def audit_variant(code,label,name,m,trainX,testX,tridx=dev,teidx=test):
        m.fit(trainX,y[tridx]);p=m.predict_proba(testX)[:,1];base=fitted[name].predict_proba(X.iloc[teidx])[:,1];yy=y[teidx];gg=groups[teidx];lo,hi=paired_ci(yy,p,base,gg,1000)
        robust.append({'code':code,'variant':label,'model':name,'n_evaluation':len(teidx),'evaluation':'fixed test','auc':roc_auc_score(yy,p),'auc_difference_from_base':roc_auc_score(yy,p)-roc_auc_score(yy,base),'lower_95':lo,'upper_95':hi,'test':'paired group bootstrap, conditional on fitted models'})
    note('Robustness checks R1 to R9')
    for name in MODELS:
        for seed in [SEED+101,SEED+102]:
            cv=StratifiedGroupKFold(3,shuffle=True,random_state=seed);vals=[]
            for tr,va in cv.split(Xd,yd,gd):
                m=make_model(name,cols,params[name]).fit(Xd.iloc[tr],yd[tr]);vals.append(roc_auc_score(yd[va],m.predict_proba(Xd.iloc[va])[:,1]))
            robust.append({'code':'R1','variant':f'development seed {seed}','model':name,'n_evaluation':len(dev),'evaluation':'development CV only','auc':np.mean(vals),'lower_95':np.nan,'upper_95':np.nan,'fold_sd':np.std(vals,ddof=1),'test':'fold SD, not confidence interval'})
    # R2 standard stratified random split, compare only clients held out under both splits.
    altdev,alttest=train_test_split(np.arange(len(df)),test_size=.2,stratify=y,random_state=SEED);common=np.intersect1d(test,alttest)
    for name in MODELS:
        audit_variant('R2','random split, common held out intersection',name,make_model(name,cols,params[name]),X.iloc[altdev],X.iloc[common],altdev,common)
    Xalt=X.copy();Xalt['EDUCATION']=Xalt.EDUCATION.replace({0:4,5:4,6:4});Xalt['MARRIAGE']=Xalt.MARRIAGE.replace({0:3})
    for c in [c for c in cols if c.startswith('PAY_') and not c.startswith('PAY_AMT')]:Xalt[c]=Xalt[c].clip(lower=0)
    Xmoney=X.copy()
    for c in [c for c in cols if c.startswith('BILL_AMT') or c.startswith('PAY_AMT')]:Xmoney[c]=np.sign(Xmoney[c])*np.log1p(np.abs(Xmoney[c]))
    for name in MODELS:
        audit_variant('R3','unknown code grouping sensitivity',name,make_model(name,cols,params[name]),Xalt.iloc[dev],Xalt.iloc[test])
        audit_variant('R3','signed log monetary variables',name,make_model(name,cols,params[name]),Xmoney.iloc[dev],Xmoney.iloc[test])
        for method in ['none','sigmoid','isotonic']:
            pp=pred[name+'__'+method].to_numpy();robust.append({'code':'R4','variant':method,'model':name,'n_evaluation':len(test),'evaluation':'fixed test','auc':roc_auc_score(yt,pp),'log_loss':log_loss(yt,np.clip(pp,1e-6,1-1e-6)),'utility_ratio5':np.mean(utility(yt,pp)),'test':'see predictive_confidence_intervals.csv'})
        audit_variant('R5','class weighted model before recalibration',name,make_model(name,cols,params[name],True),Xd,Xt)
    spline=make_model('Logistic_Spline',cols).fit(Xd,yd);splp=spline.predict_proba(Xt)[:,1];lo,hi=paired_ci(yt,splp,rawl,gt)
    robust.append({'code':'R6','variant':'logistic spline baseline','model':'Logistic_Spline','n_evaluation':len(test),'evaluation':'fixed test','auc':roc_auc_score(yt,splp),'auc_difference_from_base':roc_auc_score(yt,splp)-roc_auc_score(yt,rawl),'lower_95':lo,'upper_95':hi,'test':'paired group bootstrap versus logistic WoE'})
    Xe=X.copy();Xe['UTILIZATION']=df.BILL_AMT1/df.LIMIT_BAL.replace(0,np.nan);Xe['POSITIVE_DELINQUENCY_MONTHS']=(df[['PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']]>0).sum(axis=1);Xe['BILL_VARIABILITY']=df[[f'BILL_AMT{k}' for k in range(1,7)]].std(axis=1);Xe['PAYMENT_VARIABILITY']=df[[f'PAY_AMT{k}' for k in range(1,7)]].std(axis=1)
    assert not Xe.isna().any().any()
    for name in MODELS:audit_variant('R6','same additional behavioral proxies',name,make_model(name,list(Xe.columns),params[name]),Xe.iloc[dev],Xe.iloc[test])
    for row in decisions:robust.append({'code':'R7','variant':f'cost ratio {row["cost_ratio"]}','evaluation':'fixed test',**row,'test':'all prespecified cost scenarios, normalized utility'})
    for bseed in [SEED+130,SEED+131]:
        bg=Xd.iloc[np.random.default_rng(bseed).choice(len(dev),32,replace=False)];sv,b,r,scale=explain(fitted['XGBoost'],'XGBoost',E,bg);imp=np.abs(sv).mean(axis=0)
        robust.append({'code':'R8','variant':f'interventional background seed {bseed}','model':'XGBoost','evaluation':'fixed development profiles','rank_spearman':spearmanr(baseimports['XGBoost'][0],imp).statistic,'max_additivity_error':np.max(np.abs(b+sv.sum(axis=1)-r)),'test':'explanation sensitivity, no causal claim'})
    sv,b,r,scale=explain(fitted['XGBoost'],'XGBoost',E,path_dependent=True)
    robust.append({'code':'R8','variant':'tree path dependent, log odds','model':'XGBoost','evaluation':'fixed development profiles','rank_spearman':spearmanr(baseimports['XGBoost'][0],np.abs(sv).mean(axis=0)).statistic,'max_additivity_error':np.max(np.abs(b+sv.sum(axis=1)-r)),'test':'different feature dependence assumptions'})
    Xsex=X.copy();Xsex['SEX']=df.SEX;Xnoage=X.drop(columns='AGE')
    for name in MODELS:
        audit_variant('R9','include sex',name,make_model(name,list(Xsex.columns),params[name]),Xsex.iloc[dev],Xsex.iloc[test])
        audit_variant('R9','exclude age',name,make_model(name,list(Xnoage.columns),params[name]),Xnoage.iloc[dev],Xnoage.iloc[test])
    save(pd.DataFrame(robust),'table_07_robustness.csv')
    # Scientific figures with exported underlying coordinates.
    fig,axes=plt.subplots(1,2,figsize=(10,4.2));curve=[]
    colors=['#333333','#5f877b','#254969']
    for name,color in zip(MODELS,colors):
        p=pred[name+'__none'].to_numpy();fpr,tpr,th=roc_curve(yt,p);pr,rec,thp=precision_recall_curve(yt,p)
        axes[0].plot(fpr,tpr,label=name.replace('_',' '),color=color);axes[1].plot(rec,pr,label=name.replace('_',' '),color=color)
        for a,b in zip(fpr,tpr):curve.append({'model':name,'curve':'ROC','x':a,'y':b})
        for a,b in zip(rec,pr):curve.append({'model':name,'curve':'PR','x':a,'y':b})
    axes[0].plot([0,1],[0,1],color='gray',linestyle=':');axes[0].set(xlabel='False positive rate',ylabel='True positive rate')
    axes[1].axhline(yt.mean(),color='gray',linestyle=':');axes[1].set(xlabel='Recall',ylabel='Precision');axes[0].legend(fontsize=8);axes[1].legend(fontsize=8);fig.tight_layout();fig.savefig(FIG/'figure_01_roc_precision_recall.png');fig.savefig(FIG/'figure_01_roc_precision_recall.svg');plt.close(fig);save(pd.DataFrame(curve),'roc_pr_coordinates.csv')
    fig,axes=plt.subplots(1,2,figsize=(10,4.2));reliability=[]
    for method,color in zip(['none',calibrators['XGBoost'].method],['#7e5750','#254969']):
        p=pred['XGBoost__'+method].to_numpy();frac,mean=calibration_curve(yt,p,n_bins=10,strategy='quantile');axes[0].plot(mean,frac,'o-',label=method,color=color)
        for a,b in zip(mean,frac):reliability.append({'model':'XGBoost','calibration':method,'mean_prediction':a,'observed_rate':b})
        dd=pd.DataFrame(decisions);dd=dd[(dd.model=='XGBoost')&(dd.calibration==method)];axes[1].plot(dd.cost_ratio,dd.utility_per_client,'o-',label=method,color=color)
    axes[0].plot([0,1],[0,1],color='gray',linestyle=':');axes[0].set(xlabel='Mean predicted probability',ylabel='Observed default rate');axes[1].set(xlabel='Loss to reward ratio',ylabel='Normalized utility per client');axes[0].legend();axes[1].legend();fig.tight_layout();fig.savefig(FIG/'figure_03_calibration_decision.png');fig.savefig(FIG/'figure_03_calibration_decision.svg');plt.close(fig);save(pd.DataFrame(reliability),'calibration_coordinates.csv')
    # Independent integrity checks on exported data.
    assertions={'n_total':len(df)==30000,'default_count':sum(y)==6636,'split_complete':len(dev)+len(test)==len(df),'no_group_overlap':len(set(gd)&set(gt))==0,'test_predictions_finite':bool(np.isfinite(pred.select_dtypes('number')).all().all()),'stability_replicates':all((ss.groupby('model').size()==200)),'robustness_coverage':set(pd.DataFrame(robust).code)=={f'R{k}' for k in range(1,10)},'auc_gini_identity':all(abs(r['gini']-(2*r['auc']-1))<1e-12 for r in metricrows),'scorecard_order':bool(np.corrcoef(pred.scorecard_points,pred['Logistic_WoE__none'])[0,1]<0)}
    assert all(assertions.values()),assertions
    (OUT/'integrity_checks.json').write_text(json.dumps({k:bool(v) for k,v in assertions.items()},indent=2),encoding='utf-8');(OUT/'runtime_seconds.json').write_text(json.dumps({'elapsed_seconds':time.time()-started}),encoding='utf-8')
    note('Analysis complete; all output integrity checks passed')
    return {'metrics':pd.DataFrame(metricrows),'hypotheses':pd.DataFrame(hp),'checks':assertions}

def resume_from_saved_models():
    from joblib import load
    import sys
    sys.modules['__main__'].WoETransformer=WoETransformer
    sys.modules['__main__'].Calibrator=Calibrator
    started=time.time();df=load_data();y=df[TARGET].to_numpy().astype(int);X=df.drop(columns=['ID','SEX',TARGET]);cols=list(X.columns)
    groups=signatures(df.drop(columns=['ID',TARGET]));dev,test=next(StratifiedGroupKFold(5,shuffle=True,random_state=SEED).split(X,y,groups))
    Xd=X.iloc[dev].reset_index(drop=True);yd=y[dev];gd=groups[dev];Xt=X.iloc[test].reset_index(drop=True);yt=y[test];gt=groups[test]
    fitted={};calibrators={};params={};oofs={};oo=pd.read_csv(DATA/'development_oof_predictions.csv')
    for name in MODELS:
        obj=load(ART/f'{name}.joblib');fitted[name]=obj['model'];calibrators[name]=obj['calibrator'];params[name]=obj['params'];oofs[name]=oo[name].to_numpy()
    pred=pd.read_csv(DATA/'test_predictions.csv');sx=pred.SEX.to_numpy();px=pred['XGBoost__selected'].to_numpy();reject=px>1/6;rawl=pred['Logistic_WoE__none'].to_numpy()
    metricrows=pd.read_csv(OUT/'table_03_predictive_performance.csv').to_dict('records');hp=pd.read_csv(OUT/'table_08_hypothesis_summary.csv').to_dict('records');decisions=pd.read_csv(OUT/'table_05_decision_scenarios.csv').to_dict('records')
    old=OUT/'run_log.txt'
    if old.exists():LOG.extend(old.read_text(encoding='utf-8').splitlines())
    note('Technical resume from saved fitted models; no retuning or hypothesis changes')
    # Fixed SHAP profiles: development near threshold, plus stratified held-out explanations.
    devcal=calibrators['XGBoost'].predict(oofs['XGBoost']);evaldev=np.argsort(np.abs(devcal-1/6))[:80];E=Xd.iloc[evaldev];rng=np.random.default_rng(SEED+70)
    bgix=rng.choice(len(dev),32,replace=False);BG=Xd.iloc[bgix]
    testeval=np.sort(np.concatenate([rng.choice(np.flatnonzero(yt==k),192,replace=False) for k in [0,1]]));Et=Xt.iloc[testeval]
    explanation_rows=[];importance=[];baseimports={}
    for name in MODELS:
        note(f'Explanations {name}')
        sv,base,raw,scale=explain(fitted[name],name,Et,BG);residual=np.abs(base+sv.sum(axis=1)-raw)
        for ii,row in enumerate(sv):
            for jj,col in enumerate(cols):explanation_rows.append({'ID':int(pred.iloc[testeval[ii]].ID),'model':name,'variable':col,'contribution':row[jj],'base_value':base,'output_scale':scale,'model_output':raw[ii]})
        for col,v in zip(cols,np.abs(sv).mean(axis=0)):importance.append({'model':name,'variable':col,'mean_absolute_contribution':v,'output_scale':scale})
        svd,bd,rd,scd=explain(fitted[name],name,E,BG);baseimports[name]=(np.abs(svd).mean(axis=0),family_importance(svd,cols)[1])
        save(pd.DataFrame({'model':[name],'evaluation_n':[len(Et)],'max_additivity_error':[residual.max()],'mean_additivity_error':[residual.mean()],'output_scale':[scale]}),f'additivity_{name}.csv')
        assert residual.max()<.01,(name,residual.max())
        if name=='XGBoost':
            imp=np.abs(sv).mean(axis=0);chosen=np.argsort(imp)[-12:]
            fig,ax=plt.subplots(figsize=(7,4.8));ax.barh(np.array(cols)[chosen],imp[chosen],color='#264764');ax.set_xlabel('Mean absolute SHAP contribution in log odds');fig.tight_layout();fig.savefig(FIG/'figure_02_shap_global.png');fig.savefig(FIG/'figure_02_shap_global.svg');plt.close(fig)
            sel=np.argsort(np.abs(devcal[evaldev]-1/6))[:2];local=[]
            fig,axes=plt.subplots(1,2,figsize=(10,4.7))
            for ax,j in zip(axes,sel):
                v=svd[j];top=np.argsort(np.abs(v))[-8:];ax.barh(np.array(cols)[top],v[top],color=np.where(v[top]>0,'#a54d42','#264764'));ax.set_title(f'Development ID {int(df.iloc[dev[evaldev[j]]].ID)}');ax.set_xlabel('SHAP contribution in log odds')
                for c,vv in zip(cols,v):local.append({'ID':int(df.iloc[dev[evaldev[j]]].ID),'variable':c,'contribution':vv,'base_log_odds':bd,'predicted_pd':float(expit(rd[j])),'selection':'closest to threshold using development OOF PD'})
            fig.tight_layout();fig.savefig(FIG/'figure_02_shap_local.png');fig.savefig(FIG/'figure_02_shap_local.svg');plt.close(fig);save(pd.DataFrame(local),'shap_local_development.csv')
    save(pd.DataFrame(explanation_rows),'shap_test_contributions.csv');save(pd.DataFrame(importance),'shap_global_importance.csv')
    # 200 complete grouped bootstrap refits, fixed hyperparameters and fixed development profiles.
    note('Starting 200 grouped bootstrap refits per model for explanation stability')
    bootix=list(bootstrap_indices(gd,200,SEED+88));stability=[]
    for name in MODELS:
        original,famorig=baseimports[name];toporiginal=set(np.argsort(original)[-5:])
        def one(item):
            k,ix=item;m=make_model(name,cols,params[name]);m.named_steps['clf'].set_params(n_jobs=2) if name!='Logistic_WoE' else None
            m.fit(Xd.iloc[ix],yd[ix]);sv,b,r,sc=explain(m,name,E,BG);imp=np.abs(sv).mean(axis=0);families,fimp=family_importance(sv,cols)
            top=set(np.argsort(imp)[-5:]);return {'model':name,'replicate':k+1,'rank_spearman_variables':spearmanr(original,imp).statistic,'rank_spearman_families':spearmanr(famorig,fimp).statistic,'top5_jaccard':len(top&toporiginal)/len(top|toporiginal),'max_additivity_error':float(np.max(np.abs(b+sv.sum(axis=1)-r))),'output_scale':sc}
        with ThreadPoolExecutor(max_workers=2) as ex:
            for row in ex.map(one,enumerate(bootix)):
                stability.append(row)
                if row['replicate']%25==0:note(f'{name} explanation bootstrap {row["replicate"]}/200')
        save(pd.DataFrame(stability),'explanation_stability_bootstrap.csv')
    ss=pd.DataFrame(stability);summ=[]
    for name,frame in ss.groupby('model'):
        for met in ['rank_spearman_variables','rank_spearman_families','top5_jaccard']:
            lo,hi=np.quantile(frame[met],[.025,.975]);summ.append({'model':name,'metric':met,'mean':frame[met].mean(),'median':frame[met].median(),'lower_95_refit_distribution':lo,'upper_95_refit_distribution':hi,'replicates':len(frame)})
    save(pd.DataFrame(summ),'table_04_explanation_stability.csv')
    # Robustness checks map exactly to R1 through R9.
    robust=[]
    def audit_variant(code,label,name,m,trainX,testX,tridx=dev,teidx=test):
        m.fit(trainX,y[tridx]);p=m.predict_proba(testX)[:,1];base=fitted[name].predict_proba(X.iloc[teidx])[:,1];yy=y[teidx];gg=groups[teidx];lo,hi=paired_ci(yy,p,base,gg,1000)
        robust.append({'code':code,'variant':label,'model':name,'n_evaluation':len(teidx),'evaluation':'fixed test','auc':roc_auc_score(yy,p),'auc_difference_from_base':roc_auc_score(yy,p)-roc_auc_score(yy,base),'lower_95':lo,'upper_95':hi,'test':'paired group bootstrap, conditional on fitted models'})
    note('Robustness checks R1 to R9')
    for name in MODELS:
        for seed in [SEED+101,SEED+102]:
            cv=StratifiedGroupKFold(3,shuffle=True,random_state=seed);vals=[]
            for tr,va in cv.split(Xd,yd,gd):
                m=make_model(name,cols,params[name]).fit(Xd.iloc[tr],yd[tr]);vals.append(roc_auc_score(yd[va],m.predict_proba(Xd.iloc[va])[:,1]))
            robust.append({'code':'R1','variant':f'development seed {seed}','model':name,'n_evaluation':len(dev),'evaluation':'development CV only','auc':np.mean(vals),'lower_95':np.nan,'upper_95':np.nan,'fold_sd':np.std(vals,ddof=1),'test':'fold SD, not confidence interval'})
    # R2 standard stratified random split, compare only clients held out under both splits.
    altdev,alttest=train_test_split(np.arange(len(df)),test_size=.2,stratify=y,random_state=SEED);common=np.intersect1d(test,alttest)
    for name in MODELS:
        audit_variant('R2','random split, common held out intersection',name,make_model(name,cols,params[name]),X.iloc[altdev],X.iloc[common],altdev,common)
    Xalt=X.copy();Xalt['EDUCATION']=Xalt.EDUCATION.replace({0:4,5:4,6:4});Xalt['MARRIAGE']=Xalt.MARRIAGE.replace({0:3})
    for c in [c for c in cols if c.startswith('PAY_') and not c.startswith('PAY_AMT')]:Xalt[c]=Xalt[c].clip(lower=0)
    Xmoney=X.copy()
    for c in [c for c in cols if c.startswith('BILL_AMT') or c.startswith('PAY_AMT')]:Xmoney[c]=np.sign(Xmoney[c])*np.log1p(np.abs(Xmoney[c]))
    for name in MODELS:
        audit_variant('R3','unknown code grouping sensitivity',name,make_model(name,cols,params[name]),Xalt.iloc[dev],Xalt.iloc[test])
        audit_variant('R3','signed log monetary variables',name,make_model(name,cols,params[name]),Xmoney.iloc[dev],Xmoney.iloc[test])
        for method in ['none','sigmoid','isotonic']:
            pp=pred[name+'__'+method].to_numpy();robust.append({'code':'R4','variant':method,'model':name,'n_evaluation':len(test),'evaluation':'fixed test','auc':roc_auc_score(yt,pp),'log_loss':log_loss(yt,np.clip(pp,1e-6,1-1e-6)),'utility_ratio5':np.mean(utility(yt,pp)),'test':'see predictive_confidence_intervals.csv'})
        audit_variant('R5','class weighted model before recalibration',name,make_model(name,cols,params[name],True),Xd,Xt)
    spline=make_model('Logistic_Spline',cols).fit(Xd,yd);splp=spline.predict_proba(Xt)[:,1];lo,hi=paired_ci(yt,splp,rawl,gt)
    robust.append({'code':'R6','variant':'logistic spline baseline','model':'Logistic_Spline','n_evaluation':len(test),'evaluation':'fixed test','auc':roc_auc_score(yt,splp),'auc_difference_from_base':roc_auc_score(yt,splp)-roc_auc_score(yt,rawl),'lower_95':lo,'upper_95':hi,'test':'paired group bootstrap versus logistic WoE'})
    Xe=X.copy();Xe['UTILIZATION']=df.BILL_AMT1/df.LIMIT_BAL.replace(0,np.nan);Xe['POSITIVE_DELINQUENCY_MONTHS']=(df[['PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']]>0).sum(axis=1);Xe['BILL_VARIABILITY']=df[[f'BILL_AMT{k}' for k in range(1,7)]].std(axis=1);Xe['PAYMENT_VARIABILITY']=df[[f'PAY_AMT{k}' for k in range(1,7)]].std(axis=1)
    assert not Xe.isna().any().any()
    for name in MODELS:audit_variant('R6','same additional behavioral proxies',name,make_model(name,list(Xe.columns),params[name]),Xe.iloc[dev],Xe.iloc[test])
    for row in decisions:robust.append({'code':'R7','variant':f'cost ratio {row["cost_ratio"]}','evaluation':'fixed test',**row,'test':'all prespecified cost scenarios, normalized utility'})
    for bseed in [SEED+130,SEED+131]:
        bg=Xd.iloc[np.random.default_rng(bseed).choice(len(dev),32,replace=False)];sv,b,r,scale=explain(fitted['XGBoost'],'XGBoost',E,bg);imp=np.abs(sv).mean(axis=0)
        robust.append({'code':'R8','variant':f'interventional background seed {bseed}','model':'XGBoost','evaluation':'fixed development profiles','rank_spearman':spearmanr(baseimports['XGBoost'][0],imp).statistic,'max_additivity_error':np.max(np.abs(b+sv.sum(axis=1)-r)),'test':'explanation sensitivity, no causal claim'})
    sv,b,r,scale=explain(fitted['XGBoost'],'XGBoost',E,path_dependent=True)
    robust.append({'code':'R8','variant':'tree path dependent, log odds','model':'XGBoost','evaluation':'fixed development profiles','rank_spearman':spearmanr(baseimports['XGBoost'][0],np.abs(sv).mean(axis=0)).statistic,'max_additivity_error':np.max(np.abs(b+sv.sum(axis=1)-r)),'test':'different feature dependence assumptions'})
    Xsex=X.copy();Xsex['SEX']=df.SEX;Xnoage=X.drop(columns='AGE')
    for name in MODELS:
        audit_variant('R9','include sex',name,make_model(name,list(Xsex.columns),params[name]),Xsex.iloc[dev],Xsex.iloc[test])
        audit_variant('R9','exclude age',name,make_model(name,list(Xnoage.columns),params[name]),Xnoage.iloc[dev],Xnoage.iloc[test])
    save(pd.DataFrame(robust),'table_07_robustness.csv')
    # Scientific figures with exported underlying coordinates.
    fig,axes=plt.subplots(1,2,figsize=(10,4.2));curve=[]
    colors=['#333333','#5f877b','#254969']
    for name,color in zip(MODELS,colors):
        p=pred[name+'__none'].to_numpy();fpr,tpr,th=roc_curve(yt,p);pr,rec,thp=precision_recall_curve(yt,p)
        axes[0].plot(fpr,tpr,label=name.replace('_',' '),color=color);axes[1].plot(rec,pr,label=name.replace('_',' '),color=color)
        for a,b in zip(fpr,tpr):curve.append({'model':name,'curve':'ROC','x':a,'y':b})
        for a,b in zip(rec,pr):curve.append({'model':name,'curve':'PR','x':a,'y':b})
    axes[0].plot([0,1],[0,1],color='gray',linestyle=':');axes[0].set(xlabel='False positive rate',ylabel='True positive rate')
    axes[1].axhline(yt.mean(),color='gray',linestyle=':');axes[1].set(xlabel='Recall',ylabel='Precision');axes[0].legend(fontsize=8);axes[1].legend(fontsize=8);fig.tight_layout();fig.savefig(FIG/'figure_01_roc_precision_recall.png');fig.savefig(FIG/'figure_01_roc_precision_recall.svg');plt.close(fig);save(pd.DataFrame(curve),'roc_pr_coordinates.csv')
    fig,axes=plt.subplots(1,2,figsize=(10,4.2));reliability=[]
    for method,color in zip(['none',calibrators['XGBoost'].method],['#7e5750','#254969']):
        p=pred['XGBoost__'+method].to_numpy();frac,mean=calibration_curve(yt,p,n_bins=10,strategy='quantile');axes[0].plot(mean,frac,'o-',label=method,color=color)
        for a,b in zip(mean,frac):reliability.append({'model':'XGBoost','calibration':method,'mean_prediction':a,'observed_rate':b})
        dd=pd.DataFrame(decisions);dd=dd[(dd.model=='XGBoost')&(dd.calibration==method)];axes[1].plot(dd.cost_ratio,dd.utility_per_client,'o-',label=method,color=color)
    axes[0].plot([0,1],[0,1],color='gray',linestyle=':');axes[0].set(xlabel='Mean predicted probability',ylabel='Observed default rate');axes[1].set(xlabel='Loss to reward ratio',ylabel='Normalized utility per client');axes[0].legend();axes[1].legend();fig.tight_layout();fig.savefig(FIG/'figure_03_calibration_decision.png');fig.savefig(FIG/'figure_03_calibration_decision.svg');plt.close(fig);save(pd.DataFrame(reliability),'calibration_coordinates.csv')
    # Independent integrity checks on exported data.
    assertions={'n_total':len(df)==30000,'default_count':sum(y)==6636,'split_complete':len(dev)+len(test)==len(df),'no_group_overlap':len(set(gd)&set(gt))==0,'test_predictions_finite':bool(np.isfinite(pred.select_dtypes('number')).all().all()),'stability_replicates':all((ss.groupby('model').size()==200)),'robustness_coverage':set(pd.DataFrame(robust).code)=={f'R{k}' for k in range(1,10)},'auc_gini_identity':all(abs(r['gini']-(2*r['auc']-1))<1e-12 for r in metricrows),'scorecard_order':bool(np.corrcoef(pred.scorecard_points,pred['Logistic_WoE__none'])[0,1]<0)}
    assert all(assertions.values()),assertions
    (OUT/'integrity_checks.json').write_text(json.dumps({k:bool(v) for k,v in assertions.items()},indent=2),encoding='utf-8');(OUT/'runtime_seconds.json').write_text(json.dumps({'elapsed_seconds':time.time()-started}),encoding='utf-8')
    note('Analysis complete; all output integrity checks passed')
    return {'metrics':pd.DataFrame(metricrows),'hypotheses':pd.DataFrame(hp),'checks':assertions}

if __name__=='__main__':run()
