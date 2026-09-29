import sys,json
from credit_scoring import analysis as a
import numpy as np,pandas as pd

def finalize():
 r=a.OUT;pred=pd.read_csv(a.DATA/'test_predictions.csv');ss=pd.read_csv(r/'explanation_stability_bootstrap.csv');df=a.load_data();y=df[a.TARGET].to_numpy();groups=a.signatures(df.drop(columns=['ID',a.TARGET]));dev,test=next(a.StratifiedGroupKFold(5,shuffle=True,random_state=a.SEED).split(df,y,groups));metrics=pd.read_csv(r/'table_03_predictive_performance.csv');rb=pd.read_csv(r/'table_07_robustness.csv')
 checks={'n_total':len(df)==30000,'default_count':int(sum(y))==6636,'split_complete':len(dev)+len(test)==len(df),'no_group_overlap':len(set(groups[dev])&set(groups[test]))==0,'test_predictions_finite':bool(np.isfinite(pred.select_dtypes('number')).all().all()),'stability_replicates':bool((ss.groupby('model').size()==200).all()),'robustness_coverage':set(rb.code)=={f'R{k}' for k in range(1,10)},'auc_gini_identity':bool(np.max(np.abs(metrics.gini-(2*metrics.auc-1)))<1e-12),'scorecard_order':bool(np.corrcoef(pred.scorecard_points,pred['Logistic_WoE__none'])[0,1]<0),'additivity_all':all(pd.read_csv(r/f'additivity_{n}.csv').max_additivity_error.max()<.01 for n in a.MODELS),'weighted_recalibration_complete':len(pd.read_csv(r/'R5_weighted_recalibration.csv'))==9}
 assert all(checks.values()),checks
 (r/'integrity_checks.json').write_text(json.dumps({k:bool(v) for k,v in checks.items()},indent=2),encoding='utf-8')
 # Grouped decomposition uses absolute value only after summing related variables.
 sv=pd.read_csv(r/'shap_test_contributions.csv');sv['family']=sv.variable.map(a.group_family);family=sv.groupby(['model','ID','family','output_scale']).contribution.sum().reset_index();family['absolute_contribution']=abs(family.contribution);imp=family.groupby(['model','family','output_scale']).absolute_contribution.mean().reset_index();a.save(imp,'shap_family_importance.csv')
 fig,axes=a.plt.subplots(1,3,figsize=(12,4.8))
 for ax,name in zip(axes,a.MODELS):
  t=imp[imp.model==name].sort_values('absolute_contribution');ax.barh(t.family,t.absolute_contribution,color='#264764');ax.set_title(name.replace('_',' '));ax.set_xlabel('Mean absolute contribution\n'+str(t.output_scale.iloc[0]))
 fig.tight_layout();fig.savefig(a.FIG/'figure_02_shap_families.png');fig.savefig(a.FIG/'figure_02_shap_families.svg');a.plt.close(fig)
 # Reliability intervals resample the same predictor groups; bins fixed before resampling.
 yt=pred.y.to_numpy();gt=groups[test];methods=['none','sigmoid','isotonic'];definitions=[];rows=[]
 for method in methods:
  p=pred['XGBoost__'+method].to_numpy();edges=np.unique(np.quantile(p,np.linspace(0,1,11)));idx=np.digitize(p,edges[1:-1]);definitions.append((method,p,idx,len(edges)-1))
 boot={m:[] for m in methods}
 for ix in a.bootstrap_indices(gt,2000,a.SEED+99):
  for method,p,idx,k in definitions:
   counts=np.bincount(idx[ix],minlength=k);tot=np.bincount(idx[ix],weights=yt[ix],minlength=k);boot[method].append(np.divide(tot,counts,out=np.full(k,np.nan),where=counts>0))
 for method,p,idx,k in definitions:
  b=np.asarray(boot[method])
  for j in range(k):
   sel=idx==j
   if sel.any():rows.append({'model':'XGBoost','calibration':method,'bin':j+1,'n':int(sel.sum()),'mean_prediction':float(p[sel].mean()),'observed_rate':float(yt[sel].mean()),'lower_95':float(np.nanquantile(b[:,j],.025)),'upper_95':float(np.nanquantile(b[:,j],.975)),'scope':'fixed quantile bins; grouped test bootstrap'})
 rr=pd.DataFrame(rows);a.save(rr,'calibration_coordinates_with_intervals.csv');u=pd.read_csv(r/'decision_utility_confidence_intervals.csv');fig,axes=a.plt.subplots(1,2,figsize=(10.5,4.5))
 colors=['#333333','#648779','#aa6457']
 for method,color in zip(methods,colors):
  t=rr[rr.calibration==method];axes[0].plot(t.mean_prediction,t.observed_rate,'o-',label=method,color=color,markersize=4);axes[0].fill_between(t.mean_prediction,t.lower_95,t.upper_95,color=color,alpha=.10)
 axes[0].plot([0,1],[0,1],':',color='gray');axes[0].set(xlabel='Mean predicted probability',ylabel='Observed default rate',title='Calibration sensitivities')
 t=u[(u.model=='XGBoost')&(u.calibration=='none')].sort_values('cost_ratio');axes[1].errorbar(t.cost_ratio,t.estimate,yerr=[t.estimate-t.lower_95,t.upper_95-t.estimate],fmt='o-',capsize=3,color='#264764',label='Selected rule: no calibration');axes[1].set(xlabel='Loss to reward ratio',ylabel='Normalized utility per client',title='Prespecified cost scenarios');axes[0].legend(fontsize=8);axes[1].legend(fontsize=8);fig.tight_layout();fig.savefig(a.FIG/'figure_03_calibration_decision.png');fig.savefig(a.FIG/'figure_03_calibration_decision.svg');a.plt.close(fig)
 print('Final integrity checks',checks,flush=True)
if __name__=='__main__':finalize()
