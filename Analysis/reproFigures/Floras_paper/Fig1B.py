# %%
# general loading functions
import sys
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 

# built-in modules
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd

# Figure 1B - example visual neuron
from Admin.csv_queryExp import load_data,simplify_recdat
from Analysis.neural.utils.ev_dat import postactive
from Analysis.neural.utils.plotting import my_rasterPSTH

# load single dataset 

# selected cluster -- 
# my typical examples: 
# vis unit: 
# aud unit: 
probe = 'probe0'
recordings = load_data(
    subject = 'FT022',
    expDate = '2021-07-20',
    expNum = 1,
    data_name_dict={
        'events':{'_av_trials':'table'},
        probe:{'spikes':['times','clusters']}
    }
    )

ev,spikes,_,_ = simplify_recdat(recordings.iloc[0],probe=probe)
b,v,a,_ = postactive(ev)

cID = 5
azimuths =np.array([-90,-60,-30,0,30,60,90])  # which azimuths to plot 
sel_contrast = v.contrast.max().values
sel_spl = a.SPL.max().values

# parameters of plotting 
bin_kwargs={'tscale':[None],
            'pre_time':.03,'post_time': .2, 
            'bin_size':0.005, 'smoothing':0.02,
            'return_fr':True,'baseline_subtract':True
            }

event_kwargs = {
        'event_colors':['blue','magenta']
}

plot_kwargs = {
        'pethlw':2, 'rasterlw':2, 
        'erralpha':.4, 
        'n_rasters':30,
        'onset_marker': 'tick','onset_marker_size':10,'onset_marker_color':'grey',

}


_,ax=plt.subplots(1,azimuths.size,figsize=(5,2),sharey=True)
for idx,azi in enumerate(azimuths):
    VisOnsets = v.sel(azimuths=azi,contrast=sel_contrast).values.flatten()
    AudOnsets = a.sel(azimuths=azi,SPL=sel_spl).values.flatten()

    my_rasterPSTH(spikes.times,spikes.clusters,[VisOnsets, AudOnsets],
                    [cID],ax=ax[idx],ax1=ax[idx],include_PSTH=True,include_raster=False,
                    **bin_kwargs,**plot_kwargs,**event_kwargs)

    ax[idx].set_xlabel('%.0f deg' % azi)

plt.show()
# %%

# Fig 1C -- all the neurons plotted in SC 

