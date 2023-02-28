
import numpy as np 
from Analysis.neural.utils.spike_dat import get_binned_rasters
import matplotlib.pyplot as plt


def plot_sempsth(m,sem,t_bin,ax,errbar_kwargs={'color': 'blue', 'alpha': 0.5}):    
    #ax.fill_between(np.arange(0,m.size,1), m - sem, m + sem,**errbar_kwargs)
    # with proper timebins
    ax.fill_between(t_bin, m - sem, m + sem,**errbar_kwargs)
    #ax.plot(t_bin,m,**errbar_kwargs)

def off_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    
def off_topspines(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def off_exceptx(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_yticks([])
    ax.set_xlabel('')
    
def off_excepty(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])
    ax.set_xlabel('')

def rgb_to_hex(rgb):
    r = rgb[0]
    g = rgb[1]
    b = rgb[2]
    return '#{:02x}{:02x}{:02x}'.format(r,g,b)

def plot_the_psth(spike_times, spike_clusters, cluster_id, events, tscale, pre_time, post_time, bin_size,
                  smoothing, return_fr,pethcolor,erralpha,pethlw,ax,error_bars='sem'):

    """
    Plot psth events based on any type of binning. Input arguments are largely the same as for 
    utils.get_binned_rasters


    Inputs: 
    :param spike_times: spike times (in seconds)
    :type spike_times: array-like
    :param spike_clusters: cluster ids corresponding to each event in `spikes`
    :type spike_clusters: array-like
    :param cluster_ids: subset of cluster ids for calculating rasters
    :type cluster_ids: array-like
    :param events: times (in seconds) to align rasters to
    :type events: array-like
    :param tscale: bin_edges (in seconds) to bin rasters to. If None, tscale is caluculated from pre and post time
    :param pre_time: time (in seconds) to precede align times in raster
    :type pre_time: float
    :param post_time: time (in seconds) to follow align times in raster
    :type post_time: float
    :param bin_size: width of time windows (in seconds) to bin spikes
    :type bin_size: float
    :param smoothing: standard deviation (in seconds) of Gaussian kernel for
        smoothing rasters; use `smoothing=0` to skip smoothing
    :type smoothing: float
    :param return_fr: `True` to return (estimated) firing rate, `False` to return spike counts
    :type return_fr: bool
    :param pethcolor: color of peth plot
    :type pethcolor: str
    :param erralpha: alpha value for error bars on peth
    :type erraplha: float
    :param pethlw: linewidth of peth
    :type lethlw: float
    :param ax: axis handler for figure 
    :param error_bars: type of data to calculate error bars from. Can be 'std','sem' or 'None'
    :type error_bars: str

    :return max value of the psth, in case it is needed for determining figure limits   


    """

    binned_raster_data=get_binned_rasters(spike_times,spike_clusters,cluster_id,
                                            events,tscale,pre_time,post_time,bin_size,smoothing,return_fr)


    mean = binned_raster_data['rasters'].mean(axis=0)[0, :]

    ax.plot(binned_raster_data.tscale, mean,color=pethcolor,lw=pethlw)
    if error_bars == 'std':
        bars = binned_raster_data['rasters'].std(axis=0)[0, :]

    elif error_bars == 'sem':
        bars = binned_raster_data['rasters'].std(axis=0)[0, :] / np.sqrt(len(events))

    else:
        bars = np.zeros_like(mean)

    if error_bars != 'none':
        ax.fill_between(binned_raster_data.tscale, mean - bars, mean + bars, color=pethcolor,alpha=erralpha)
    
    lim = [(mean.min() + bars[mean.argmin()]),(mean.max() + bars[mean.argmax()])] 

    ax.set_xlim([-pre_time, post_time])
    ax.set_ylabel('Firing Rate' if return_fr else 'Number of spikes')

    return lim

def load_default_params_raster():
    bin_kwargs={'tscale':[None],
            'pre_time':.4,'post_time': .5, 
            'bin_size':0.005, 'smoothing':0.02,
            'return_fr':True 
            }

    event_kwargs = {
            'event_colors':['blue']
    }
    plot_kwargs = {
            'pethlw':2, 'rasterlw':2, 
            'erralpha':.4, 
            'n_rasters':20,
            'onset_marker': 'tick','onset_marker_size':10,'onset_marker_color':'grey',

    }

    return bin_kwargs,event_kwargs,plot_kwargs
    
def my_rasterPSTH(spike_times,  # Spike times first
                  spike_clusters,  # Then cluster ids
                  events,
                  cluster_id,  # Identity of the cluster we plot
                  pre_time=0.25, post_time=0.25,tscale=[None],  # Time before and after the event, or array of timescale
                  error_bars='sem',  # Whether we want Stdev, SEM, or no error
                  include_PSTH=True,
                  include_raster=True,  # adds a raster to the bottom
                  n_rasters=100, # How many raster traces to include per event set
                  bin_size=0.025,
                  smoothing=0.005,
                  return_fr=True,        
                  event_colors=['blue','magenta','green'],
                  pethlw=2,onset_marker='line',onset_marker_size=2,onset_marker_color='k',
                  erralpha=0.5, 
                  rasterlw=2, 
                  plot_edge=None,
                  ax=None,
                  ax1=None): # this is to set the ratio with which you make PSTHs bigger than mean
   

    if ax is None:
        if (include_raster==True)&(include_PSTH==True): 
            fig, (ax, ax1)=plt.subplots(2, 1, sharex=True)
        
        elif (include_raster==True)&(include_PSTH==False):
            plt.figure()
            ax1 = plt.gca()          
        else: 
            plt.figure()
            ax = plt.gca()
        


    #gr=10
    
    trialcount=0
    for i,evt in enumerate(events):       
        if include_PSTH:
            lims=plot_the_psth(spike_times, spike_clusters, [cluster_id], evt,tscale, pre_time, post_time,
                            bin_size,smoothing, return_fr,event_colors[i],erralpha,pethlw,ax,error_bars)
            p1=lims[1] *1.05 

            if i>0: 
                if p1>pmax: 
                    pmax=p1
            else: 
                pmax=p1
            
        trialcount+=evt.size

    

    # Plot the event marker line. 
    if include_PSTH:
        ax.set_xlim([-pre_time, post_time])    
        if plot_edge==None: 
            plot_edge=pmax
        ax.vlines(0., 0., plot_edge, color='black', alpha=0.5)
        #ax.set_ylim([ 0., plot_edge])
        ax.set_yticks([0., plot_edge/2, plot_edge])
        ax.set_ylabel('Firing Rate' if return_fr else 'Number of spikes')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    

    # Move the x axis line from the bottom of the plotting space to zero if including a raster,
    # Then plot the raster

    if include_raster:
        tickheight =10
        
        clu_spks = spike_times[spike_clusters == cluster_id]
        
        if n_rasters is None: 
            tickedges = np.arange(0., -trialcount*tickheight - 1e-5, -tickheight)
        else:
            tickedges = np.arange(0., -n_rasters*len(events)*tickheight - 1e-5, -tickheight)
        ct=0 # counts how many trials one has plotted overall
        for e,evtgrp in enumerate(events):
            if n_rasters is None:
                n_rasters = evtgrp.size
            for i, t in enumerate(evtgrp[:n_rasters]):
                l=i+ct
                idx = np.bitwise_and(clu_spks >= t - pre_time, clu_spks <= t + post_time)
                event_spks = clu_spks[idx]
                ax1.vlines(event_spks - t, tickedges[l + 1], tickedges[l],color=event_colors[e],lw=rasterlw)
            ct+=i # add 
            #print(ct,i, trialcount)



        if include_PSTH:
            # set the labels of the PSTH 
            ax.axhline(0., color='black')
            plt.setp(ax.get_xticklabels(), visible=False)
            #ax.set_xticks([])
            ax.spines['bottom'].set_visible(False)
            ax.set_xlabel('')
        elif 'line' in onset_marker:
            ax1.vlines(0,tickedges[ct],tickedges[0],'grey',lw=onset_marker_size,color=onset_marker_color)
        else:
            ax1.plot(0,0,marker='v',markersize=onset_marker_size,color=onset_marker_color)

        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.set_xlabel('Time (s) after event')
        plt.setp(ax1.get_yticklabels(), visible=False)

    else:
        ax.set_xlabel('Time (s) after event')  

def plot_cluster_location(clusInfo,sel_clus_idx=None,ax=None): 
    if ax is None: 
        _,ax = plt.subplots(1,1,figsize=(2,10))
    # parameters for 4-shank probe (can adapt...)    
    ax.vlines(20,0,2880*2,'k',alpha=.1)
    ax.vlines(220,0,2880*2,'k',alpha=.1)
    ax.vlines(420,0,2880*2,'k',alpha=.1)
    ax.vlines(620,0,2880*2,'k',alpha=.1)

    ax.scatter(clusInfo.XPos.values,clusInfo.Depth.values,s=4,color='grey',alpha=.5)

    # plot selected 
    # sel_clus_idx =1 
    if sel_clus_idx is not None: 
        ax.scatter(clusInfo.XPos.values[sel_clus_idx],clusInfo.Depth.values[sel_clus_idx],s=15,color='red',alpha=1,linewidths=15)

    off_topspines(ax)
    ax.set_xlabel('xpos')
    ax.set_ylabel('depth')

def plot_cluster_waveform(clusInfo,sel_clus_idx=1,ax=None):
    raw_waveform = clusInfo.rawWaveforms[sel_clus_idx]['spkMapMean']
    #max_chan = clusInfo.maxChannels[sel_clus_idx]
    max_chan = np.argmax(np.max((np.abs(raw_waveform-np.tile(raw_waveform[:,0],(raw_waveform.shape[1],1)).T)),axis=1))
    # if it is off number get things down to the even index just so I can plot consistently 
    if (max_chan % 2) != 0: 
        max_chan=max_chan-1


    tot_show=30
    max_amp=np.max(raw_waveform[max_chan])
    min_amp=np.min(raw_waveform[max_chan])
    start = int(max_chan-tot_show/2)
    end = int(max_chan+tot_show/2)
    if start<0: 
        start=0
        end=tot_show 
    if end>raw_waveform.shape[0]:
        start= raw_waveform.shape[0]-tot_show
        end = raw_waveform.shape[0]

    selected_all=raw_waveform[start:end,:]

    if ax is None:
        fig,ax = plt.subplots(1,3,figsize=(9,3))
        fig.patch.set_facecolor('xkcd:white')

        
    ax[0].plot(raw_waveform[max_chan,:]) # could put texts on this plot about amplitude etc.
    ax[1].imshow(selected_all[1::2,:],aspect='auto',cmap='coolwarm',vmin=min_amp*1.02,vmax=max_amp*1.02)
    ax[2].imshow(selected_all[0::2,:],aspect='auto',cmap='coolwarm',vmin=min_amp*1.02,vmax=max_amp*1.02)
    ax[1].set_title('left_channels')
    ax[2].set_title('right_channels')

    [off_axes(ax[i]) for i in range(3)]

def share_lim(ax,dim='y'): 
    """
    share x lim for all subplots in ax 
    Parameters: 
    ------------
    ax: numpy ndarray of matplotlib.pyplot AxesSubplots   
    dim: str
        dimension of arrays to be shared
        can be in pronciple x,y, or xy       
    """
    if 'x' in dim:
        lims = np.array([myax.get_ylim() for _,myax in np.ndenumerate(ax)])
        shared_lims = (np.min(lims[:,0]),np.max(lims[:,1]))
        [myax.set_ylim(shared_lims) for _,myax in np.ndenumerate(ax)]        

    if 'y' in dim:
        lims = np.array([myax.get_ylim() for _,myax in np.ndenumerate(ax)])
        shared_lims = (np.min(lims[:,0]),np.max(lims[:,1]))
        [myax.set_ylim(shared_lims) for _,myax in np.ndenumerate(ax)]


def plot_raster_withmovementonset(spikes,leftchoicestims,rightchoicestims,choicemove_times,neuronID,ax,t_before=.2,t_after=1):
    
    plot_kwargs = {
        'pre_time':t_before,
        'post_time':t_after,
        'include_PSTH':False,
        'include_raster':True,
        'n_rasters':leftchoicestims.size+rightchoicestims.size,
        'ax':ax,
        'ax1':ax
    }  
    
    my_rasterPSTH(spikes.times,  # Spike times first
                    spikes.clusters,  # Then cluster ids
                    [leftchoicestims,
                    rightchoicestims],
                    neuronID,  # Identity of the cluster we plot 
                    event_colors=['blue','red'],rasterlw=1,**plot_kwargs)

    
    my_rasterPSTH(choicemove_times,  # Spike times first
                    np.ones(choicemove_times.size).astype('int'),  # Then cluster ids
                    [leftchoicestims,
                    rightchoicestims],
                    1,  # Identity of the cluster we plot,
                    event_colors=['black','black'],rasterlw=3,**plot_kwargs)
    
    off_axes(ax) 
    ax.set_ylim([-1000,10])


def plot_driftMap(spikes,ax=None):
    nColorBins = 40 
    #ampRange = np.quantile(spikes.amps,(0.1,0.9))
    #colorBins = np.linspace(ampRange[0],ampRange[1],nColorBins)
    colorBins = np.linspace(0,0.0005,nColorBins)
    colors = plt.cm.Greys(np.linspace(0.1,1,nColorBins))

    if not ax: 
        _,ax = plt.subplots(1,1,figsize=(10,10))

    for b in range(nColorBins-1):
        idx = (spikes.amps>=colorBins[b]) & (spikes.amps<colorBins[b+1])
        ax.plot(spikes.times[idx],spikes.depths[idx],'.',color=colors[b+1],markersize=1)