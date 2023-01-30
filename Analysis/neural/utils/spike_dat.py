import numpy as np
from scipy.signal import convolve, gaussian
from scipy.stats import zscore
import warnings # just because of some pandas thing in 
warnings.filterwarnings("ignore")

from Admin.csv_queryExp import Bunch

def get_binned_rasters(spike_times, spike_clusters, cluster_ids, align_times, tscale=[None],
 pre_time=0.2,post_time=0.5, bin_size=0.025, smoothing=0.025, return_fr=True,baseline_subtract=False):
    """
    Bin spikes to create rasters around events

    :param spike_times: spike times (in seconds)
    :type spike_times: array-like
    :param spike_clusters: cluster ids corresponding to each event in `spikes`
    :type spike_clusters: array-like
    :param cluster_ids: subset of cluster ids for calculating rasters
    :type cluster_ids: array-like
    :param align_times: times (in seconds) to align rasters to
    :type align_times: array-like
    :param tscale: bin_edges (in seconds) to bin rasters to. If None, tscale is caluculated from pre and post time
    :type align_times: array-like
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
    :return: rasters
    :rtype: rasters: Bunch({'rasters': binned_spikes_, 'tscale': ts, 'cscale': ids})
    """

    # initialize containers
    

     # compute floating tscale if not supplied - in this case smoothing can be requested
    if tscale[0]==None:
        n_offset = 5 * int(np.ceil(smoothing / bin_size))  # get rid of boundary effects for smoothing
        n_bins_pre = int(np.ceil(pre_time / bin_size)) + n_offset
        n_bins_post = int(np.ceil(post_time / bin_size)) + n_offset
        n_bins = n_bins_pre + n_bins_post
        tscale = np.arange(-n_bins_pre, n_bins_post + 1) * bin_size

        # caluate the s
        total_trange=[np.min(align_times) - (n_bins_pre + 1) * bin_size,
                    np.max(align_times) + (n_bins_post + 1) * bin_size]
    else:
    # if floating tscale is supplied smoothing cannot be requested    
        smoothing=0
        n_bins = tscale.size -1 # -1 as it represents bin edges?
        total_trange = [np.min(align_times),np.max(align_times)]

    binned_spikes = np.zeros(shape=(len(align_times), len(cluster_ids), n_bins))

    # build gaussian kernel if requested
    if smoothing > 0:
        w = n_bins - 1 if n_bins % 2 == 0 else n_bins
        window = gaussian(w, std=smoothing / bin_size)
        # half (causal) gaussian filter
        window[:int(np.ceil(w/2))] = 0
        window /= np.sum(window)
        binned_spikes_conv = np.copy(binned_spikes)

    ids = np.unique(cluster_ids)

    # filter spikes outside of the loop
    idxs = np.bitwise_and(spike_times >= total_trange[0],
                          spike_times <= total_trange[1])
    idxs = np.bitwise_and(idxs, np.isin(spike_clusters, cluster_ids))
    spike_times = spike_times[idxs]
    spike_clusters = spike_clusters[idxs]

   
    #determine whether tscale is even or uneven (different indexing)  
    bin_sizes=np.diff(tscale)
    unique_bin_size = np.unique((bin_sizes/bin_sizes[0]).round(decimals=1))*bin_sizes[0]
    
    # bin spikes
    for i, t_0 in enumerate(align_times):
        # define bin edges
        ts = tscale + t_0
        # filter spikes
        idxs = np.bitwise_and(spike_times >= ts[0], spike_times <= ts[-1])
        i_spikes = spike_times[idxs]
        i_clusters = spike_clusters[idxs]

        # bin spikes similar to bincount2D: x = spike times, y = spike clusters
        xscale = ts
        # if the bins are evenly spaced, one can just divide by bin size to get bin index
        if unique_bin_size.size==1:     
            xind = (np.floor((i_spikes - np.min(ts)) / bin_sizes[0])).astype(np.int64)
        else: 
            # if not then the binning is a bit more intensive 
            
            rel_spike_times = i_spikes - t_0
            rel_spike_times = np.repeat(rel_spike_times[:,np.newaxis],tscale.size,axis=1)
            # we subtract the bin edges and check which one is greater than 0 
            # subtracting 1 because we start the indexing at 0 (1st bin edge only = 0th index)
            xind  = (np.sum((rel_spike_times-tscale)>0,axis=1)-1).astype(np.int64)   

        yscale, yind = np.unique(i_clusters, return_inverse=True)
        nx, ny = [xscale.size, yscale.size]
        ind2d = np.ravel_multi_index(np.c_[yind, xind].transpose(), dims=(ny, nx))
        r = np.bincount(ind2d, minlength=nx * ny, weights=None).reshape(ny, nx)

        # store (ts represent bin edges, so there are one fewer bins)
        bs_idxs = np.isin(ids, yscale)
        binned_spikes[i, bs_idxs, :] = r[:, :-1]

        # smooth
        if smoothing > 0:
            idxs = np.where(bs_idxs)[0]
            for j in range(r.shape[0]):
                binned_spikes_conv[i, idxs[j], :] = convolve(
                    r[j, :], window, mode='same', method='auto')[:-1]

    # average
    if smoothing > 0:
        binned_spikes_ = np.copy(binned_spikes_conv)
    else:
        binned_spikes_ = np.copy(binned_spikes)

    if return_fr:
        # to account also for uneven binsizes
        FRbins=np.repeat(np.repeat(bin_sizes[np.newaxis,np.newaxis,:],len(cluster_ids),axis=1),len(align_times),axis=0)
        binned_spikes_ /= FRbins
        #binned_spikes_ /= bin_size


    if smoothing > 0:
         binned_spikes_ = binned_spikes_[:, :, n_offset:-n_offset]
         tscale = tscale[n_offset:-n_offset]

    # package output
    tscale = (tscale[:-1] + tscale[1:]) / 2

    if baseline_subtract:
        # subtract the mean baseline (i.e. the mean before 0 on the tscale)
        baseline = (binned_spikes_[:,:, tscale<0]).mean(axis=2)
        baseline = np.tile(baseline[:,:,np.newaxis],binned_spikes_.shape[2])
        binned_spikes_= binned_spikes_-baseline

    rasters = Bunch({'rasters': binned_spikes_, 'tscale': tscale, 'cscale': ids})
    
    return rasters

def bin_spikes_pos_and_time(spikes,depth_corr_window_spacing=40,spike_binning_t=.01):

    depth_corr_window = 0 # MUA window in microns
    max_depths = 5760

    depth_corr_bins =np.arange(0,max_depths-depth_corr_window_spacing+1,depth_corr_window_spacing) # add one for even arange

    # also determine which shank things are on 
    shank_bins=np.arange(0,4,1) # unified for 4shank recordings.
    spike_binning_t_edges = np.arange(np.nanmin(spikes.times),np.nanmax(spikes.times)+spike_binning_t,spike_binning_t)
    binned_spikes_depth = np.zeros((shank_bins.size,depth_corr_bins.size,spike_binning_t_edges.size-1));


    for i in shank_bins:
        for j in range(depth_corr_bins.size):
            curr_depth_templates_idx = np.where((spikes._av_shankIDs==i) &                                                
                                                (spikes.depths >= depth_corr_bins[j]) &
                                                (spikes.depths < depth_corr_bins[j]+depth_corr_window_spacing))[0]

            N=np.histogram(spikes.times[curr_depth_templates_idx],spike_binning_t_edges)    
            binned_spikes_depth[i,j,:]=N[0]

    
    return Bunch({'array': binned_spikes_depth,'xposscale':shank_bins,'depthscale':depth_corr_bins,'tscale':spike_binning_t_edges})

def call_bombcell_params():
    # select units based on metrics 
    metric_thresholds = {
        "max_peaks": 2,
        "max_throughs":1,
        "is_somatic":1, 
        "min_spatial_decay_slope":-20,
        "min_waveform_duration": 100,
        "max_waveform_duration": 800,
        "max_waveform_baseline_fraction": .3,
        "max_percentage_spikes_missing":20,
        "min_spike_num":300,
        "max_refractory_period_violations":10,
        "min_amp":10,
    }

    return metric_thresholds
    
def bombcell_sort_units(clusdat,max_peaks,max_throughs,
                        is_somatic,min_spatial_decay_slope,
                        min_waveform_duration,max_waveform_duration,
                        max_waveform_baseline_fraction,max_percentage_spikes_missing,
                        min_spike_num,max_refractory_period_violations,min_amp):

    #any unit that is not discarded as noise or selected as well isolated is mua.
    #maybe there ought to be an option to classify well isolated axonal units...
    clusdat['bombcell_class']='mua'

    # assign noise 

    ix = ~ (
        (clusdat.nPeaks>max_peaks) | 
        (clusdat.nTroughs>max_throughs) |
        (clusdat.somatic>=is_somatic) | 
        (clusdat.spatialDecaySlope<=min_spatial_decay_slope) |
        (clusdat.waveformDuration<min_waveform_duration) |
        (clusdat.waveformDuration>max_waveform_duration) |
        (clusdat.waveformBaseline>=max_waveform_baseline_fraction)
    ) 

    clusdat['bombcell_class'][ix]='noise'

    # assign well isolated units 
    ix = (
        (clusdat.bombcell_class != 'noise') &
        (clusdat.Spknum>min_spike_num) &
        (clusdat.Fp <= max_refractory_period_violations) &
        (clusdat.rawAmplitude>min_amp) &
        (clusdat.percSpikesMissing>max_percentage_spikes_missing)
    )

    clusdat['bombcell_class'][ix]='good'


    return clusdat

def bincount2D(x, y, xbin=0, ybin=0, xlim=None, ylim=None, weights=None,xsmoothing=0):
    """
    Computes a 2D histogram by aggregating values in a 2D array.
    :param x: values to bin along the 2nd dimension (c-contiguous)
    :param y: values to bin along the 1st dimension
    :param xbin:
        scalar: bin size along 2nd dimension
        0: aggregate according to unique values
        array: aggregate according to exact values (count reduce operation)
    :param ybin:
        scalar: bin size along 1st dimension
        0: aggregate according to unique values
        array: aggregate according to exact values (count reduce operation)
    :param xlim: (optional) 2 values (array or list) that restrict range along 2nd dimension
    :param ylim: (optional) 2 values (array or list) that restrict range along 1st dimension
    :param weights: (optional) defaults to None, weights to apply to each value for aggregation
    :param xsmoothing: (optional) smoothing along the x axis with a half-gaussian, with sigma given by this value
    :return: 3 numpy arrays MAP [ny,nx] image, xscale [nx], yscale [ny]
    """
    # if no bounds provided, use min/max of vectors
    if xlim is None:
        xlim = [np.min(x), np.max(x)]
    if ylim is None:
        ylim = [np.min(y), np.max(y)]

    def _get_scale_and_indices(v, bin, lim):
        # if bin is a nonzero scalar, this is a bin size: create scale and indices
        if np.isscalar(bin) and bin != 0:
            scale = np.arange(lim[0], lim[1] + bin / 2, bin)
            ind = (np.floor((v - lim[0]) / bin)).astype(np.int64)
        # if bin == 0, aggregate over unique values
        else:
            scale, ind = np.unique(v, return_inverse=True)
        return scale, ind

    xscale, xind = _get_scale_and_indices(x, xbin, xlim)
    yscale, yind = _get_scale_and_indices(y, ybin, ylim)
    # aggregate by using bincount on absolute indices for a 2d array
    nx, ny = [xscale.size, yscale.size]
    ind2d = np.ravel_multi_index(np.c_[yind, xind].transpose(), dims=(ny, nx))
    r = np.bincount(ind2d, minlength=nx * ny, weights=weights).reshape(ny, nx)

    # if a set of specific values is requested output an array matching the scale dimensions
    if not np.isscalar(xbin) and xbin.size > 1:
        _, iout, ir = np.intersect1d(xbin, xscale, return_indices=True)
        _r = r.copy()
        r = np.zeros((ny, xbin.size))
        r[:, iout] = _r[:, ir]
        xscale = xbin

    if not np.isscalar(ybin) and ybin.size > 1:
        _, iout, ir = np.intersect1d(ybin, yscale, return_indices=True)
        _r = r.copy()
        r = np.zeros((ybin.size, r.shape[1]))
        r[iout, :] = _r[ir, :]
        yscale = ybin

    if xsmoothing>0: 
        w = xscale.size #[tscale.size - 1 if tscale.size % 2 == 0 else tscale.size]
        window = gaussian(w, std=xsmoothing / xbin)
        # half (causal) gaussian filter
        window[:int(np.ceil(w/2))] = 0
        window /= np.sum(window) # 
        binned_spikes_conv = [convolve(r[j, :], window, mode='same', method='auto')[:,np.newaxis] for j in range(r.shape[0])]#[:-1]
        r = np.concatenate(binned_spikes_conv,axis=1).T

    return r, xscale, yscale

def cross_correlation(A, B, zscorea=True, zscoreb=True):
    '''Compute correlation for each column of A against
    every column of B (e.g. B is predictions).
    Parameters
    ----------
    A : 2D np.ndarray (n, p)
    B : 2D np.ndarray (n, q)
    Returns
    -------
    cross_corr : 2D np.ndarray (p, q)
    '''
    n = A.shape[0]

    # If needed
    if zscorea: A = zscore(A)
    if zscoreb: B = zscore(B)
    corr = np.dot(A.T, B)/float(n)
    return corr