import sys,os,json,time
from credit_scoring import analysis as a
from joblib import load
import numpy as np,pandas as pd
sys.modules['__main__'].WoETransformer=a.WoETransformer;sys.modules['__main__'].Calibrator=a.Calibrator

def run_supplement():
 df=a.load_data();y=df[a.TARGET].to_numpy().astype(int);X=df.drop(columns=['ID','SEX',a.TARGET]);cols=list(X.columns);groups=a.signatures(df.drop(columns=['ID',a.TARGET]));dev,test=a.load_split_indices(df);Xd=X.iloc[dev].reset_index(drop=True);yd=y[dev];gd=groups[dev];Xt=X.iloc[test].reset_index(drop=True);yt=y[test];gt=groups[test];pred=pd.read_csv(a.DATA/'test_predictions.csv');rows=[]
 for name in a.MODELS:
  print('Weighted recalibration',name,flush=True);obj=load(a.ART/f'{name}.joblib');par=obj['params'];oof=np.empty(len(dev))
  for tr,va in a.StratifiedGroupKFold(3,shuffle=True,random_state=a.SEED+170).split(Xd,yd,gd):
   m=a.make_model(name,cols,par,True).fit(Xd.iloc[tr],yd[tr]);oof[va]=m.predict_proba(Xd.iloc[va])[:,1]
  m=a.make_model(name,cols,par,True).fit(Xd,yd);raw=m.predict_proba(Xt)[:,1];chosen,cv=a.choose_calibration(oof,yd,gd,name)
  for method in ['none','sigmoid','isotonic']:
   p=a.Calibrator(method).fit(oof,yd).predict(raw);rows.append({'code':'R5','model':name,'calibration':method,'selected_on_development':method==chosen.method,**a.metrics(yt,p),'utility_ratio_5':np.mean(a.utility(yt,p)),'evaluation':'fixed test; weighting and recalibration fitted only on development'})
 a.save(pd.DataFrame(rows),'R5_weighted_recalibration.csv')
 # Alternative age boundaries are exploratory sensitivity, not new confirmatory hypotheses.
 age=pd.cut(pred.AGE,[20,34,49,np.inf],labels=['21 to 34','35 to 49','50 and above']).astype(str).to_numpy();out=[]
 for name in a.MODELS:
  p=pred[name+'__selected'].to_numpy();r=p>1/6
  for label in sorted(set(age)):
   ix=age==label;good=ix&(yt==0);bad=ix&(yt==1)
   out.append({'code':'R9','model':name,'age_group':label,'n':sum(ix),'nondefaults':sum(good),'defaults':sum(bad),'fpr':np.mean(r[good]),'tpr':np.mean(r[bad]),'acceptance_rate':np.mean(~r[ix]),'status':'exploratory alternative age boundaries'})
 a.save(pd.DataFrame(out),'R9_alternative_age_groups.csv')
 scenarios=[];vectors=[]
 for name in a.MODELS:
  obj=load(a.ART/f'{name}.joblib')
  for method in dict.fromkeys(['none',obj['calibrator'].method]):
   p=pred[name+'__'+method].to_numpy()
   for ratio in [2,5,10,20]:scenarios.append({'model':name,'calibration':method,'cost_ratio':ratio,'estimate':float(np.mean(a.utility(yt,p,ratio)))});vectors.append(a.utility(yt,p,ratio))
 vectors=np.vstack(vectors);bs=np.asarray([vectors[:,ix].mean(axis=1) for ix in a.bootstrap_indices(gt,2000,a.SEED+99)])
 for k,row in enumerate(scenarios):row.update({'lower_95':np.quantile(bs[:,k],.025),'upper_95':np.quantile(bs[:,k],.975),'replicates':2000,'scope':'conditional on fitted models and assumed costs'})
 a.save(pd.DataFrame(scenarios),'decision_utility_confidence_intervals.csv');print('Supplement complete',flush=True)
if __name__=='__main__':run_supplement()
