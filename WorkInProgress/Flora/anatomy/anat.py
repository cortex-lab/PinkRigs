
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\Audiovisual") 
# ONE loader from the PinkRig Pipeline

import utils.data_manager as dat
from utils.plotting import plot_driftMap,off_axes
from utils.io import add_PinkRigs_to_path
add_PinkRigs_to_path()
from Admin.csv_queryExp import load_data,get_recorded_channel_position
from Analysis.pyhist.assign_clusters_to_atlas import call_for_anatmap_recordings

def plot_anatmaps_longitudinal_(subject='AV025',probe = 'probe0',savefigs=False):
    """
    function to plot anatomy/drift maps of a given subject over time 
    # this uses all the spontaneuous and sparseNoise

    Parameters: 
    -----------
    mysubject: str
        subject name
    probe: str
        probe0/1
    savefigs: bool
        whether to save fig,default path is in subject/histology/driftMaps
    """

    data_dict = {
    probe:{'spikes':'all'},
    '%s_raw' % probe:{'channels':'all'},
    }

    sn_recs=load_data(data_name_dict=data_dict,subject=subject,expDef='sparseNoise')
    spont_recs=load_data(data_name_dict=data_dict,subject=subject,expDef='spontaneous')
    all_rec = pd.concat((sn_recs,spont_recs))

    savefigs = True
    # keep recordings with successful 
    is_sorted = [bool(rec[probe].spikes) for _,rec in all_rec.iterrows()]
    all_rec= all_rec[is_sorted]
    all_rec = all_rec[['Subject','expDate','expNum','expFolder',probe,'%s_raw' % probe]]  

    shank_range,depth_range = zip(*[dat.get_recorded_channel_position(rec['%s_raw' % probe].channels) for _,rec in all_rec.iterrows()])

    all_rec = all_rec.assign(
        shank_range = shank_range,
        depth_range = depth_range
    )
    # throw away recordings that are not single shank 
    is_single_shank = [(rec.shank_range[1] - rec.shank_range[0])<35 for _,rec in all_rec.iterrows()]
    all_rec = all_rec[is_single_shank]

    # calculate all combinations that exists
    depths = all_rec.depth_range.unique()
    shanks = all_rec.shank_range.unique()

    shanks_x,depths_x = np.meshgrid(shanks,depths)
    shanks_x,depths_x = np.ravel(shanks_x),np.ravel(depths_x)

    # generate figures
    for myloc in range(shanks_x.size):
        same_range = all_rec[(all_rec.shank_range==shanks_x[myloc]) & (all_rec.depth_range == depths_x[myloc])]

        fig,ax = plt.subplots(1,same_range.shape[0],figsize=(20,4),sharey=True)
        fig.patch.set_facecolor('xkcd:white')
        for cax,(_,rec) in zip(ax,same_range.iterrows()):
            spikes = rec[probe].spikes
            plot_driftMap(spikes,ax = cax)    
            off_axes(cax)
            cax.set_title(rec.expDate)
            
        plt.suptitle(
            '%s,shank %.0f,depth range: %s um' % (rec.Subject,spikes._av_shankIDs[0],depths_x[myloc])
        )
        plt.show()
        if savefigs:
            # prep path   
            savepath = (Path(rec.expFolder)).parents[1]
            savepath = savepath / 'histology/driftMaps'
            savepath.mkdir(parents=True,exist_ok=True) 

            namestring = '%s_%s_shank%.0f_botrow%.0f.png' % (
                rec.Subject,
                probe,
                spikes._av_shankIDs[0],
                depths_x[myloc][0]/15  # as 15 um is the spacing
            )
            # save 
            fig.savefig(
                (savepath / namestring), 
                dpi = 300,
                bbox_inches = 'tight'
                )


def call_locations_per_depth_spacing(subject='AV008',probe='probe0',depth_spacing=60,max_depths=5760):
    recdat = call_for_anatmap_recordings(subject=subject,probe=probe,depth_selection='auto') 

    # load the clusters for all of them
    data_dict = {
    ('%s_raw' % probe):{'clusters':['brainLocationIds_ccf_2017','depths'],'channels':'all'}
    }
    recs = []
    for _,rec in recdat.iterrows():
        f = load_data(data_name_dict=data_dict,subject=rec.Subject,expDate=rec.expDate,expNum=rec.expNum)
        recs.append(f)

    recdat = pd.concat(recs)

    shank_range,depth_range = zip(*[get_recorded_channel_position(rec[('%s_raw' % probe)].channels) for _,rec in recdat.iterrows()])
    # should also check whether the brain region id really exists. 

    recdat = recdat.assign(
        shank_range = shank_range, 
        depth_range = depth_range
    )
    recdat = recdat.assign(
        shank = [int(sh[0]/200) for sh in recdat.shank_range]
    )


    depth_bins =np.arange(0,max_depths-depth_spacing+1,depth_spacing)

    def get_region_id_per_depth(clusters,depth_bins):
        region_id_ = []
        for d in range(depth_bins.size-1):
            region_id = clusters.brainLocationIds_ccf_2017[
                (clusters.depths>=depth_bins[d]) & 
                (clusters.depths<depth_bins[d+1])
                ]
            if len(region_id)==0:
                region_id= 0 # 0 is void
            else:
                region_id = region_id[0]
            
            region_id_.append(region_id)
        
        return region_id_


    region_ids = [get_region_id_per_depth(rec['%s_raw' % probe].clusters,depth_bins) for _,rec in recdat.iterrows()]

    recdat = recdat.assign( 
        region_ids = region_ids
    )

    return recdat