"""Compact publication layouts; coordinates are inches at 5.5-inch paper width."""
import re
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

INK, GRAY = '#161923', '#8190A5'


def style_figure(fig, stem):
    if stem.startswith(('11_','12_')):
        stem='05_count_geometry'
    elif stem.startswith(('13_','14_')):
        stem='10_count_geometry_3d'
    # Equal plotting heights within each figure; compact titles and shared legends.
    grid = [(.49,2.32,2.12,1.12), (3.26,2.32,2.12,1.12),
            (.49,.49,2.12,1.12), (3.26,.49,2.12,1.12)]
    specs = {
        '01_training_and_behavior': (2.32,
            [(.46,.59,1.35,1.27),(2.27,.59,1.35,1.27),(4.08,.59,1.35,1.27)],
            ['A. Non-thinking loss','B. Thinking loss','C. Answer accuracy']),
        '02_retrieval_ablation': (3.78,grid,
            ['A. Non-thinking: answer','B. Non-thinking: numeric output',
             'C. Thinking: answer','D. Thinking: complete trace']),
        '03_head_functions_and_transport': (2.38,
            [(1.13,.56,1.19,1.52),(2.50,.56,1.19,1.52),(4.16,.56,1.22,1.52)],
            ['A. Final answer','B. Complete trace','C. Value patching']),
        '04_count_readability': (2.16,
            [(.49,.60,2.12,1.25),(3.26,.60,2.12,1.25)],
            ['A. Running index','B. Final count']),
        '05_count_geometry': (5.04,
            [(.50,3.02,2.10,1.72),(3.25,3.02,2.10,1.72),
             (.50,.66,2.10,1.72),(3.25,.66,2.10,1.72)],None),
        '06_progress_state': (2.71,
            [(1.03,.81,1.59,1.57),(3.77,.81,1.61,1.57)],
            ['A. Next-marker sources','B. Donor continuation']),
        '07_answer_readout': (2.52,
            [(1.03,.68,1.15,1.42),(2.73,.68,1.15,1.42),(4.24,.68,1.15,1.42)],
            ['A. Answer sources','B. State transplant','C. Count direction']),
        '08_learning_dynamics': (3.78,grid,
            ['A. Running-index readability','B. Final-count readability',
             'C. Top-2 ablation damage','D. Targeted value transport']),
        '09_retrieval_roles': (5.43,
            [(.44,3.08,1.74,1.92),(3.18,3.08,1.74,1.92),
             (.44,.47,1.74,1.92),(3.18,.47,1.74,1.92)],
            ['A. Non-thinking: broad','B. Thinking: broad',
             'C. Thinking: targeted','D. Thinking: successor']),
        '10_count_geometry_3d': (4.82,
            [(.12,2.72,2.00,1.75),(2.88,2.72,2.00,1.75),
             (.12,.62,2.00,1.75),(2.88,.62,2.00,1.75)],None),
    }
    height, rectangles, titles = specs[stem]
    if stem in ['01_training_and_behavior','03_head_functions_and_transport','04_count_readability','07_answer_readout']:
        reserve=.13 if stem=='03_head_functions_and_transport' else .10
        height+=reserve
        rectangles=[(x,y+reserve,w,h) for x,y,w,h in rectangles]
    if stem in ['05_count_geometry','10_count_geometry_3d']:
        height+=.13
        rectangles=[(x,y+.13,w,h) for x,y,w,h in rectangles]
    fig.set_size_inches(5.5,height)
    axes = fig.axes[:len(rectangles)]
    for i,(ax,(x,y,w,h)) in enumerate(zip(axes,rectangles)):
        ax.set_position([x/5.5,y/height,w/5.5,h/height])
        title = titles[i] if titles else ax.get_title(loc='left')
        title = re.sub(r'^\(([a-z])\) ',lambda m:m[1].upper()+'. ',title)
        title = title.replace(': Running index, ',': running, ').replace(': Final count, ',': final, ')
        title = title.replace(': running\n',': running, ').replace(': final\n',': final, ').replace('\n',', ')
        if stem in ['05_count_geometry','10_count_geometry_3d']:
            title=title.replace('Non-thinking','NT').replace('Thinking','T')
        ax.set_title(title,loc='left',fontsize=9.5,pad=6)
        ax.tick_params(labelsize=8.5,pad=2)
        ax.xaxis.label.set(fontsize=9)
        ax.yaxis.label.set(fontsize=9)
        ax.xaxis.labelpad=3
        ax.yaxis.labelpad=3
        for t in ax.texts:
            t.set_fontsize(8.5)
        if hasattr(ax,'zaxis'):
            ax.zaxis.label.set_fontsize(8.5)
            ax.tick_params(labelsize=8,pad=0)
        rename = {'Number of ablated heads':'Ablated heads', 'Number of patched heads':'Patched heads',
                  'Free-generation accuracy (%)':'Accuracy (%)',
                  'Balanced NCC accuracy (%)':'NCC accuracy (%)',
                  'Donor-count adoption (%)':'Donor adoption (%)',
                  'Control minus selected accuracy (pp)':'Extra damage (pp)',
                  'Patched minus damaged logit margin':'Margin gain'}
        ax.set_xlabel(rename.get(ax.get_xlabel(),ax.get_xlabel()))
        ax.set_ylabel(rename.get(ax.get_ylabel(),ax.get_ylabel()))
        if ax.get_xlabel()=='Training step':
            ax.set_xticks([0,200,1000,10000] if ax.get_xscale()=='symlog' else [0,5000,10000])
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    for legend in list(fig.legends):
        legend.remove()

    def legend(ax, labels=None, loc='best', **kwargs):
        handles,names=ax.get_legend_handles_labels()
        return ax.legend(handles, labels or names, loc=loc, fontsize=8.5,
                         frameon=True, facecolor='white', edgecolor='none', framealpha=1,
                         borderpad=.2, labelspacing=.25, handlelength=1.3,
                         handletextpad=.4, **kwargs)

    def footer(handles, labels, x=.5, y=.012, cols=2):
        return fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(x,y),
                          ncol=cols,fontsize=8.5,frameon=False,handlelength=1.3,
                          handletextpad=.4,columnspacing=1.1,labelspacing=.25)

    if stem=='01_training_and_behavior':
        axes[0].set_ylabel('Token loss')
        axes[1].set_ylabel('Token loss')
        axes[0].yaxis.set_major_locator(MaxNLocator(4))
        axes[1].yaxis.set_major_locator(MaxNLocator(4))
        footer([Line2D([],[],color=INK),Line2D([],[],color=INK,ls='--')],
               ['Train region','Validation region'],x=.365)
        legend(axes[2],loc='center')
    elif stem=='02_retrieval_ablation':
        for i,ax in enumerate(axes):
            legend(ax,['Selected heads','Matched controls'],loc=['upper right','center right','lower left','lower left'][i])
    elif stem=='03_head_functions_and_transport':
        axes[0].set_yticks(range(9),['Clean','Targeted Top-2','Successor L2H3',
                                    'Succ. control 1','Succ. control 2','Succ. control 3',
                                    'Broad Top-2','Targeted + succ.','Targeted + broad'])
        for ax in axes[:2]:
            ax.set(xlim=(0,118),xticks=[0,50,100])
            for t in ax.texts:
                x,y=t.get_position()
                t.set_position((x+5,y))
            for c in ax.collections:
                c.set_sizes([18])
        axes[2].set_ylabel('Logit margin')
        footer(axes[2].get_legend_handles_labels()[0],['Selected','Control','Clean','Damaged'],x=.59,cols=4)
    elif stem=='04_count_readability':
        for ax in axes:
            ax.set_ylim(-4,110)
        h,_=axes[0].get_legend_handles_labels()
        footer(h,['Non-thinking','Thinking','Chance (10%)'],cols=3)
    elif stem=='05_count_geometry':
        for ax in axes:
            ax.set_xlabel(ax.get_xlabel().replace(' discovery variance',''))
            ax.xaxis.set_major_locator(MaxNLocator(3))
            ax.yaxis.set_major_locator(MaxNLocator(3))
            # Preserve equal data units, with matching panel frames.
            ax.set_aspect('equal',adjustable='datalim')
        fig.delaxes(fig.axes[4])
        cax=fig.add_axes([.17,.31/height,.70,.065/height])
        cb=fig.colorbar(axes[0].collections[0],cax=cax,orientation='horizontal',ticks=range(1,11))
        cb.set_label('Count or running index',fontsize=9,labelpad=2)
        cax.tick_params(labelsize=8.5,pad=2,length=2)
    elif stem=='06_progress_state':
        axes[0].set_yticks(range(5),['Prompt targets','Full history','Most recent','Except recent','First half'])
        axes[0].set(ylim=(4.5,-.5),xticks=[0,50,100],xlabel='Next-marker accuracy (%)')
        footer(axes[0].get_legend_handles_labels()[0],['Removed','Control','Clean (96%)'],x=.30,y=.006,cols=2)
        axes[1].set_yticks(range(4),['Next marker\n(26 pairs)','2 markers\n(30 pairs)',
                                   '3 markers\n(26 pairs)','4 markers\n(20 pairs)'])
        axes[1].set(ylim=(3.5,-.5),xticks=[0,25,50],xlabel='Donor effect (pp)')
        footer(axes[1].get_legend_handles_labels()[0],['Full item (2 tokens)','Endpoint (1 token)'],x=.815,y=.006,cols=1)
    elif stem=='07_answer_readout':
        axes[0].set_yticks(range(5),['Clean','Prompt targets\nremoved','Prompt-budget\ncontrol',
                                   'Trace removed','Trace-budget\ncontrol'])
        axes[0].set(xlabel='Answer accuracy (%)',xticks=[0,50,100])
        axes[2].set_ylabel('')
        footer(axes[0].get_legend_handles_labels()[0],['Non-thinking (NT)','Thinking (T)'],x=.56)
        legend(axes[2],['NT control','NT','T control','T'],loc='upper left')
    elif stem=='08_learning_dynamics':
        for i,ax in enumerate(axes):
            labels=[s.replace('Non-thinking','NT').replace('Thinking','T') for s in ax.get_legend_handles_labels()[1]]
            legend(ax,labels,loc=['upper right','center right','lower left','upper left'][i])
            # Avoid cutting off endpoint markers at steps 0 and 10,000.
            ax.set_xlim(-20,11500)
    elif stem=='09_retrieval_roles':
        heads=[(l,h) for l in range(1,5) for h in range(8)]
        for i,ax in enumerate(axes):
            ax.set_yticks(range(0,32,2),[f'L{l}H{h}' for l,h in heads[::2]])
            ax.tick_params(axis='y',labelsize=8,length=0,pad=2)
            cax=fig.axes[4+i]
            x,y,w,h=rectangles[i]
            cax.set_axes_locator(None)
            cax.set_box_aspect(None)
            cax.set_aspect('auto')
            cax.set_position([(x+w+.055)/5.5,y/height,.060/5.5,h/height])
            cax.tick_params(labelsize=8,pad=2,length=2)
            cax.yaxis.label.set_fontsize(8.5)
            cax.yaxis.labelpad=3
            if i==3:
                cax.set_ylabel('Successor attention',fontsize=8.5)
    elif stem=='10_count_geometry_3d':
        # Short axis labels eliminate perspective-dependent label spill.
        for ax in axes:
            for j,axis in enumerate([ax.xaxis,ax.yaxis,ax.zaxis],1):
                axis.set_label_text('')
                axis.labelpad=0
            # Explicit 2D label anchors avoid Matplotlib's incomplete 3D tight bbox.
            ax.text2D(.23,-.055,'PC1',transform=ax.transAxes,ha='center',fontsize=8.5)
            ax.text2D(.90,.025,'PC2',transform=ax.transAxes,ha='center',fontsize=8.5)
            ax.text2D(1.055,.54,'PC3',transform=ax.transAxes,ha='left',fontsize=8.5)
            ax.set_box_aspect(ax._box_aspect,zoom=.88)
        cax=fig.axes[4]
        cax.set_position([.17,.31/height,.70,.065/height])
        cax.tick_params(labelsize=8.5,pad=2,length=2)
        cax.xaxis.label.set(fontsize=9)
        cax.xaxis.labelpad=2
    for ax in fig.axes:
        for axis in [ax.xaxis,ax.yaxis]+([ax.zaxis] if hasattr(ax,'zaxis') else []):
            lo,hi=sorted(axis.get_view_interval())
            axis.set_ticks([x for x in axis.get_majorticklocs() if lo-1e-8<=x<=hi+1e-8])
