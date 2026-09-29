from pathlib import Path
import json,numpy as np,pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
WORK=Path(r'C:\Users\ASUS\Documents\ChatGPT\灾害信息处理\scipilot_redraw_v3_20260928');REP=Path(r'D:\landslide_unet_project\reports');content=json.loads((WORK/'manuscript_content.json').read_text(encoding='utf-8'))
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica'],'font.size':6.3,'axes.titlesize':6.8,'axes.labelsize':6.2,'xtick.labelsize':5.7,'ytick.labelsize':5.7,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.5,'xtick.major.width':.5,'ytick.major.width':.5,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.dpi':600})
INK='#1F2933';MUTED='#5F6B76';BLUE='#2166AC';RED='#B2182B';GREEN='#1B7837';ORANGE='#D88700';LIGHT='#D9DEE3';GREY='#A8B0B7'
x3=pd.read_csv(REP/'q2_x3_component_summary.csv');seed=pd.read_csv(REP/'q2_x3_seed_stratified_bootstrap'/'q2_x3_seed_stratified_t_interval.csv')
models=['ResUNet','Bottleneck-LiteASK'];colors={'ResUNet':RED,'Bottleneck-LiteASK':GREEN};markers={'ResUNet':'o','Bottleneck-LiteASK':'s'}
fig=plt.figure(figsize=(183/25.4,54/25.4));gs=fig.add_gridspec(1,4,width_ratios=[1.12,.95,.95,1.02],wspace=.40);axa=fig.add_subplot(gs[0]);axb=fig.add_subplot(gs[1]);axc=fig.add_subplot(gs[2]);axd=fig.add_subplot(gs[3])
endpoints=[('delta_iou','iou_low','iou_high','IoU','iou'),('delta_balanced_iou','balanced_iou_low','balanced_iou_high','Balanced IoU','balanced_iou'),('delta_mcc','mcc_low','mcc_high','MCC','mcc')];regions=['Hokkaido','Lombok','Palu']
for k,model in enumerate(models):
 for j,(d,l,h,label,key) in enumerate(endpoints):
  sr=x3[(x3.model==model)&(x3.region=='target_macro')].iloc[0];rr=seed[(seed.architecture==model)&(seed.endpoint==key)].iloc[0];y=j+(k-.5)*.16;cl=colors[model]
  axa.hlines(y-.105,float(sr[l]),float(sr[h]),color=GREY,lw=.65,zorder=2)
  axa.hlines(y,float(rr.ci95_low),float(rr.ci95_high),color=cl,lw=1.15,zorder=4)
  axa.vlines([float(rr.ci95_low),float(rr.ci95_high)],y-.055,y+.055,color=cl,lw=.6,zorder=4)
  axa.scatter(float(rr.mean_delta),y,marker=markers[model],s=11,facecolor=cl,edgecolor='white',linewidth=.3,zorder=6)
axa.axvline(0,color=MUTED,lw=.55,ls='--');axa.set_yticks(range(3),[e[3] for e in endpoints]);axa.invert_yaxis();axa.set_xlim(-.04,.32);axa.set_xlabel('Δ (pooled − placebo)');axa.set_title('a  Target-macro endpoints',loc='left',fontweight='bold',pad=3)
for k,model in enumerate(models):
 vals=[];lo=[];hi=[]
 for reg in regions:
  r=x3[(x3.model==model)&(x3.region_label==reg)].iloc[0];vals.append(float(r.delta_iou));lo.append(float(r.iou_low));hi.append(float(r.iou_high))
 vals=np.array(vals);lo=np.array(lo);hi=np.array(hi);y=np.arange(3)+(k-.5)*.13
 axb.errorbar(vals,y,xerr=np.vstack([vals-lo,hi-vals]),fmt=markers[model],ms=3.0,mew=.5,color=colors[model],ecolor=colors[model],elinewidth=.6,capsize=1.3,zorder=3)
axb.axvline(0,color=MUTED,lw=.55,ls='--');axb.set_yticks(range(3),regions);axb.invert_yaxis();axb.set_xlim(-.025,.23);axb.set_xlabel('Δ IoU');axb.set_title('b  Region-level effects',loc='left',fontweight='bold',pad=3)
tab=content['tables']['9']['rows'];mapnames=['ResUNet','SegFormer-B0'];arms=['128','retrained 256','128 weights at 256'];arm_labels=['128→128','256→256','128 w@256'];cols=['#D9D9D9','#969696','#525252'];target={m:[] for m in mapnames};source={m:[] for m in mapnames}
for m in mapnames:
 for arm in arms:
  row=next(r for r in tab[1:] if r[0]==m and r[1]==arm);target[m].append(float(row[2]));source[m].append(float(row[3]))
x=np.arange(2);width=.22
for j,arm in enumerate(arms):
 vals=[target[m][j] for m in mapnames];axc.bar(x+(j-1)*width,vals,width,color=cols[j],edgecolor=INK,linewidth=.4,zorder=2,label=arm_labels[j])
 for xx,v in zip(x+(j-1)*width,vals):axc.text(xx,v+.006,f'{v:.3f}',ha='center',fontsize=4.7)
axc.set_xticks(x,mapnames);axc.set_ylim(0,.225);axc.set_ylabel('Target macro IoU');axc.set_title('c  Resolution target',loc='left',fontweight='bold',pad=3)
for j,arm in enumerate(arms):
 vals=[source[m][j] for m in mapnames];axd.bar(x+(j-1)*width,vals,width,color=cols[j],edgecolor=INK,linewidth=.4,zorder=2)
 for xx,v in zip(x+(j-1)*width,vals):axd.text(xx,v+.012,f'{v:.3f}',ha='center',fontsize=4.7)
axd.set_xticks(x,mapnames);axd.set_ylim(0,.9);axd.set_ylabel('Source-validation IoU');axd.set_title('d  Source validation',loc='left',fontweight='bold',pad=3)
for m_i,m in enumerate(mapnames):
 x0=m_i+width;v=source[m][2];axd.annotate(f'collapse\n{v:.3f}',xy=(x0,v),xytext=(x0,v+.17),ha='center',va='bottom',fontsize=4.7,color=RED,arrowprops=dict(arrowstyle='-|>',color=RED,lw=.55))
for ax in [axa,axb,axc,axd]:ax.grid(False);ax.tick_params(pad=1.2)
fig.subplots_adjust(left=.065,right=.995,bottom=.30,top=.86)
from matplotlib.lines import Line2D
mh=[Line2D([0],[0],color=colors[m],marker=markers[m],ms=3.2,lw=1.2,label=m) for m in models]
ah=[Line2D([0],[0],color=cols[j],lw=5,label=arm_labels[j]) for j in range(3)]
fig.legend(handles=mh,loc='lower center',bbox_to_anchor=(0.28,0.015),ncol=2,fontsize=5.0,frameon=False,handletextpad=.4,columnspacing=1.4)
fig.legend(handles=ah,loc='lower center',bbox_to_anchor=(0.77,0.015),ncol=3,fontsize=5.0,frameon=False,handletextpad=.4,columnspacing=1.2)
fig.text(0.28,0.085,'colour: seed-stratified t; grey: component',ha='center',fontsize=4.8,color=MUTED)
out=WORK/'Figure4_seed_stratified_final';fig.savefig(str(out)+'.svg',bbox_inches='tight',facecolor='white');fig.savefig(str(out)+'.pdf',bbox_inches='tight',facecolor='white');fig.savefig(str(out)+'.png',bbox_inches='tight',facecolor='white',dpi=600);plt.close(fig);print(out)
