# %% 
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

from dat_utils import get_paths
from plot_utils import get_region_colors,off_axes,copy_svg_to_clipboard

import warnings
warnings.filterwarnings('ignore')

colors = get_region_colors()
plt.rcParams.update({'font.size': 18})



colors = {'SC': '#D392C0',
 'Frontal': '#024E38',
 'Vis': '#5EA7DC',
 'Parietal': '#28A3B7',
 'Lateral': '#00BCD1'}

_, _, savepath = get_paths(r'opto\region_comparison\bi')
data = pd.read_csv(savepath / 'summary.csv')
# we fit +ve so I flip

if 'bi' in savepath.__str__():
    data['vis'] = data['visR'] - data['visL']
    data['aud'] = data['audR'] - data['audL']
    data['vis_opto'] = data['visR_opto'] - data['visL_opto']
    data['aud_opto'] = data['audR_opto'] - data['audL_opto']
    params = ['vis','aud','bias']

elif 'uni' in savepath.__str__():
    params = ['visR','visL','audR','audL','bias']

params_opto = [('%s_opto' % p) for p in params]
for p,p_opto in zip(params,params_opto):
    data['tot_%s' % p] = data[p] + data[p_opto]
    data['diff_%s' % p] = np.log(data['tot_%s' % p]/data[p])


# %%

# to compare between regions
plotted_var = ['%s_opto' % p if p=='bias'else 'diff_%s' % p for p in params]

n_plots = len(plotted_var)
fig = plt.figure(figsize=(n_plots*1.5,4))
gs = GridSpec(1,n_plots,figure=fig)
ax = np.array([fig.add_subplot(gs[0,i]) for i in range(n_plots)])
[ax[i].sharey(ax[i+1]) for i in range(n_plots-3)]

for i,p in enumerate(plotted_var):    
    sns.stripplot(data,
    x='brain_region',
    y=p,
    hue='brain_region',
    order = ['SC'],
    palette = colors,
    size=9,
    edgecolor='k',linewidth=1,
    jitter = 0,
    legend=None,ax=ax[i])

    ax[i].set_title(p)
    ax[i].axhline(0,color='k',linestyle='--')

    if 'diff' in p:
        ax[i].set_ylim([-2.5,1])
        
        if i>3:
            ax[i].set_ylabel('')
            off_axes(ax[i],which = 'exceptx')
        else:
            ax[i].set_ylabel('log(opto/control)')
            off_axes(ax[i],which = 'top')

    else:
        ax[i].set_ylabel('opto-control')
        ax[i].set_ylim([-1,4.5])
        off_axes(ax[i],which = 'top')

    ax[i].tick_params(axis='x', rotation=90)

#plt.tight_layout()


from plot_utils import copy_svg_to_clipboard,off_axes
plt.tight_layout()
plt.show()

copy_svg_to_clipboard(fig)
# %%
# control vs opto against each other
fig,ax=plt.subplots(1,len(params),
                    figsize=(12,2),sharex=True,sharey=True)

data_ = data[data.brain_region=='SC']
for i,p in enumerate(params): 
    ctrl,opto = p,('tot_%s' % p)
    sns.scatterplot(data_,x = ctrl,
                    y=opto,
                    color = 'cyan',edgecolor='k',s=50,linewidth=1,
                    ax=ax[i])
    
#     t_stat, p_value = stats.ttest_rel(params[c], params[o])


    ax[i].axline([-4,-4],[6,6],color='k',linestyle='--')
    ax[i].axhline(0,color='k',linestyle='--')
    ax[i].axvline(0,color='k',linestyle='--')
    ax[i].set_title(p)
    ax[i].set_xlabel(ctrl)
    ax[i].set_ylabel(opto)
    ax[i].legend([])

# %%

# per brain region
plt.rcParams.update({'font.size': 18})


data_long  =  data.melt(
    id_vars = ['brain_region'], 
    value_vars =['diff_vis', 'diff_aud', 'bias_opto'],
    var_name =  'param_name',
    value_name = 'delta_opto'
)
fig, ax = plt.subplots(1,1,figsize=(7,5))

sns.barplot(data = data_long,
             x = 'param_name', 
             y = 'delta_opto', 
             hue = 'brain_region', 
             hue_order = ['SC','Frontal','Vis'],
             palette = colors,
           #  errorbar= ('ci',95),
             ax = ax
             )

ax.axhline(0,color='k',linestyle='--')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()

plt.ylim([-2.5,3.5])
off_axes(ax,'top')
plt.show()
copy_svg_to_clipboard(fig)

# %%
