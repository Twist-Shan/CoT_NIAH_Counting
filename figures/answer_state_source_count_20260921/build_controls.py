"""Plot different-count and control source-count match rates on the same scale."""
from pathlib import Path
import csv
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args=parser.parse_args()
rows=list(csv.DictReader((args.data/'answer_source_count_curves.csv').open(encoding='utf-8')))
OUT=args.output; OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'Times New Roman','mathtext.fontset':'stix','font.size':9,
    'axes.labelsize':10,'axes.titlesize':11,'xtick.labelsize':9,'ytick.labelsize':9,
    'axes.linewidth':.7,'pdf.fonttype':42,'svg.fonttype':'none','legend.frameon':False})
fig=plt.figure(figsize=(6.5,2.05))
for i,(model,name,color,last) in enumerate([('Qwen3-8B','Qwen','#168DCA',36),('Gemma4-E4B','Gemma','#E87824',42)]):
    ax=fig.add_axes([[.10,.61][i],.23,.365,.57])
    for cond,label,ls,c in [('donor_transport','Different count','-',color),('same_count_seed','Same count','--',color),('self_patch','Self patch',':','#8190A5')]:
        rr=[r for r in rows if r['model']==model and r['condition']==cond]
        x=[int(r['layer']) for r in rr]; y=[float(r['mean']) for r in rr]
        ax.plot(x,y,color=c,ls=ls,lw=1.5,label=label)
        ax.fill_between(x,[float(r['ci95_low']) for r in rr],[float(r['ci95_high']) for r in rr],color=c,alpha=.14,lw=0)
    ax.set(title=f'{"AB"[i]}. {name}',xlabel='Intervention layer',ylabel='Source-count match rate',
           xlim=(1,last),ylim=(-.04,1.07),xticks=list(range(1,last+1,10)),yticks=[0,.5,1])
    ax.set_title(f'{"AB"[i]}. {name}',loc='left'); ax.set_title('')
    ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',color='#E7E8EE',lw=.6)
    ax.legend(loc='upper left',fontsize=8,handlelength=1.6,borderaxespad=.2,labelspacing=.15)
for ext in ('pdf','svg','png'):
    fig.savefig(OUT/f'nonthinking_answer_patching.{ext}',dpi=220)
print('Wrote source-count matching controls')
