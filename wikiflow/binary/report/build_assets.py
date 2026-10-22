from pathlib import Path
import csv,json,hashlib,statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent
src=OUT/'evidence'
rows=list(csv.DictReader((src/'monthly_selected.csv').open()))
summary=list(csv.DictReader((src/'summary_selected.csv').open()))
S={r['method']:r for r in summary}
methods=[('volume','Volume'),('cs_momentum','Momentum'),('binary_log13','Original Log13'),('binary_missing_0.01','Logistic + missing'),('binary_dynamics_hgb_d2_l21','HGB classifier'),('binary','LightGBM classifier'),('pairwise_expanded_0.01_a0.25','Linear pairwise + rule'),('rank50_base','Binary numeric Rank50')]
variants=[('rank50_missing','+ text availability'),('rank50_svd','+ TF-IDF SVD8 and indicator'),('rank50_lda','+ LDA8 and indicator'),('rank50_intro','+ MiniLM PCA8 and indicator')]
for m,n in methods+variants:
 rr=[r for r in rows if r['method']==m];assert len(rr)==24
 for field,field2 in [('binary_NDCG','macro_binary_NDCG'),('AP','macro_AP'),('Precision','macro_Precision')]:
  vals=[float(r[field]) for r in rr if r[field] not in ('','nan','None')]
  assert abs(statistics.mean(vals)-float(S[m][field2]))<1e-12
 assert sum(int(r['TP']) for r in rr)==int(S[m]['TP'])
 assert abs(sum(int(r['TP']) for r in rr)/sum(int(r['events']) for r in rr)-float(S[m]['micro_Recall']))<1e-12
labels=[('Volume','Current count'),('Momentum','Growth margin'),('Logistic','Log13 reference'),('Logistic','Log13 + text availability'),('HGB','Numeric + dynamics'),('LightGBM','Compact numeric'),('Linear pairwise','Compact numeric + rule blend'),('LambdaRank','Compact numeric, Rank50')]
with (OUT/'results_rows.tex').open('w') as f:
 for (m,_),(algorithm,features) in zip(methods,labels):
  r=S[m]
  f.write(algorithm+' & '+features+' & '+' & '.join(f"{float(r[k]):.4f}" for k in ['macro_binary_NDCG','macro_Precision','micro_Recall','macro_AP'])+r" \\"+'\n')
ablation=[('rank50_base','None (numeric only)'),('rank50_missing','Text availability only (control)'),('rank50_svd','Word patterns (TF-IDF / SVD)'),('rank50_lda','Topic mixture (LDA)'),('rank50_intro','Semantic meaning (MiniLM / PCA)')]
with (OUT/'variant_rows.tex').open('w') as f:
 for m,n in ablation:
  value=float(S[m]['macro_binary_NDCG']);delta=value-float(S['rank50_base']['macro_binary_NDCG'])
  f.write(n+' & '+f"{value:.4f} & {delta:+.4f}"+r" \\"+'\n')
base=[r for r in rows if r['method']=='volume'];assert sum(int(r['N']) for r in base)==5387;assert sum(int(r['events']) for r in base)==190
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'font.family':'DejaVu Sans'})
fig,ax=plt.subplots(figsize=(8,2.8));x=list(range(24));rates=[100*int(r['events'])/int(r['N']) for r in base]
ax.bar(x,rates,color='#287c8e');ax.axhline(100*190/5387,color='#b55a2c',ls='--',label='Pooled: 190 / 5,387 = 3.53%')
ax.set_xticks(x[::3],[r['month'] for r in base][::3],rotation=25);ax.set_ylabel('Known-label surge rate (%)');ax.set_ylim(0,max(rates)*1.3);ax.legend(frameon=False,fontsize=10);fig.tight_layout();fig.savefig(OUT/'figures/class_balance.pdf');plt.close(fig)
D={m:{r['month']:r for r in rows if r['method']==m} for m,_ in methods}
deltas=[]
for r in base:
 mo=r['month'];a=D['rank50_base'][mo]['binary_NDCG'];b=D['cs_momentum'][mo]['binary_NDCG'];deltas.append(None if not a or a=='nan' else float(a)-float(b))
fig,ax=plt.subplots(figsize=(8,3))
plot_deltas=[0.0 if v is None else v for v in deltas]
ax.bar(x,plot_deltas,color=['#287c8e' if v>=0 else '#bd6646' for v in plot_deltas])
ax.axhline(0,color='#334455',lw=.8)
ax.scatter([7],[0],color='#287c8e',s=20,zorder=3)
ticks=sorted((set(x[::3])-{6})|{7});ax.set_xticks(ticks,[base[i]['month'] for i in ticks],rotation=35,ha='right')
ax.set_ylabel('NDCG@50 difference')
fig.tight_layout();fig.savefig(OUT/'figures/monthly_delta.pdf');plt.close(fig)
(OUT/'evidence/monthly_selected.csv').write_text('')
with (OUT/'evidence/monthly_selected.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n');w.writeheader();w.writerows([r for r in rows if r['method'] in dict(methods+variants)])
with (OUT/'evidence/summary_selected.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=summary[0].keys(),lineterminator='\n');w.writeheader();w.writerows([S[m] for m,n in methods+variants])
loss=sorted([(v,base[i]['month']) for i,v in enumerate(deltas) if v is not None])
print('worst',loss[:5]);print('wins ties losses',sum(v>1e-12 for v in deltas if v is not None),sum(abs(v)<=1e-12 for v in deltas if v is not None),sum(v < -1e-12 for v in deltas if v is not None))
print('Verified 12 configurations; no fits, raw data or holdout access.')
