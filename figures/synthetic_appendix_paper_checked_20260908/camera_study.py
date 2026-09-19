"""Inspect camera angles without changing data, bases, or coordinate scaling."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
DATA = OUT.parent/'synthetic_appendix_aurora/plot_data/11_pca_common_l2_points.csv'
CAMERAS = [(20,-35),(22,-65),(22,-115),(25,35),(25,65),(15,135)]
CMAP = LinearSegmentedColormap.from_list('aurora',['#23165C','#6750E8','#00C2FF','#39E58C','#F6E36A'])

def draw(ax,d,camera,small=False):
    xyz=d[['pc1','pc2','pc3']].to_numpy()
    ax.set_proj_type('ortho')
    ax.view_init(elev=camera[0],azim=camera[1])
    ax.scatter(*xyz.T,c=d.occurrence,cmap=CMAP,vmin=1,vmax=10,
               s=7 if len(d)>100 else 14,alpha=.48,depthshade=False,rasterized=True)
    centers=d.groupby('occurrence')[['pc1','pc2','pc3']].mean()
    ax.plot(*centers.to_numpy().T,color='#8190A5',lw=.8,zorder=5)
    ax.scatter(*centers.to_numpy().T,c=centers.index,cmap=CMAP,vmin=1,vmax=10,
               s=30,edgecolors='#161923',linewidths=.5,depthshade=False,zorder=6)
    spans=np.ptp(xyz,axis=0)
    for i,(setlim,axis) in enumerate([(ax.set_xlim,ax.xaxis),(ax.set_ylim,ax.yaxis),(ax.set_zlim,ax.zaxis)]):
        setlim(xyz[:,i].min()-.08*spans[i],xyz[:,i].max()+.08*spans[i])
        axis.set_major_locator(MaxNLocator(2))
        axis.pane.fill=False
        axis._axinfo['grid'].update(color='#E2E4EB',linewidth=.45)
    ax.set_box_aspect(spans,zoom=.91)
    ax.tick_params(labelsize=7 if small else 9,pad=0)
    ax.set_xlabel('PC1',labelpad=-3 if small else 0)
    ax.set_ylabel('PC2',labelpad=-3 if small else 0)
    ax.set_zlabel('PC3',labelpad=-3 if small else 0)

if __name__=='__main__':
    (OUT/'qa').mkdir(exist_ok=True)
    d=pd.read_csv(DATA)
    d=d[d['mode']=='thinking']
    plt.rcParams.update({'font.family':'Times New Roman','font.size':9})
    fig=plt.figure(figsize=(15,5.8))
    for j,cam in enumerate(CAMERAS):
        for i,answer in enumerate([False,True]):
            ax=fig.add_subplot(2,6,i*6+j+1,projection='3d',computed_zorder=False)
            draw(ax,d[d.endpoint.str.contains('answer_query').eq(answer)],cam,True)
            ax.set_title(f'T {"final" if answer else "running"}: {cam}',fontsize=10)
    fig.subplots_adjust(left=.01,right=.97,bottom=.05,top=.94,wspace=.04,hspace=.12)
    fig.savefig(OUT/'qa/camera_study.png',dpi=150,bbox_inches='tight')
