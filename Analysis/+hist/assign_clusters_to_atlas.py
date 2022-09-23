import json,re
import numpy as np
from ibllib.atlas import AllenAtlas
atlas = AllenAtlas(25)

def get_chan_coordinates(root):
    """
    function to get coordinates on npix and allen atlas for a given channel 
    Parameters: 
    -----------
    root: Pathlib.path
        directory of ibl_format file 

    Returns: 
    --------
    chan_IDs
    chan_pos: numpy ndarray
        relative lateral channel position on neuropixel 
        0th dim: x, 1st dim: y
    allencoords_xyz: numpy ndarray
        relative channel posititon on the allen atlas in xyz reference frame
    region_ID: list
        region ID that the channel is in. Can be used for hierarchical serarches in the atlas.
    region_acronym: list
        region name that the channel is in. 

    """
    channel_locations = open(root / 'channel_locations.json',)
    channel_locations = json.load(channel_locations)
    chan_names = np.array(list(channel_locations.keys()))
    chan_names = chan_names[['channel' in ch for ch in chan_names]] #get the channels only
    #chan_IDs = [int(re.split('_',ch)[-1]) for ch in chan_names]
    allenx  = [channel_locations[ch]['x'] for ch in chan_names] 
    alleny  = [channel_locations[ch]['y'] for ch in chan_names] 
    allenz  = [channel_locations[ch]['z'] for ch in chan_names] 
    regionID = [channel_locations[ch]['brain_region_id'] for ch in chan_names] 
    region_acronym = [channel_locations[ch]['brain_region'] for ch in chan_names]
    chan_pos_x = [channel_locations[ch]['lateral'] for ch in chan_names] 
    chan_pos_y = [channel_locations[ch]['axial'] for ch in chan_names] 

    chan_pos = np.array([chan_pos_x, chan_pos_y]).T
    allencoords_xyz=np.array([allenx,alleny,allenz]).T


    return chan_pos,allencoords_xyz,regionID,region_acronym


def coordinate_matching(local_coordinate_array,target_coordinate_array):
    """
    performs a coordinate matching bsed on xy positions. 
    Basically outputs which indices in target coordinate array match the coordinates in local
    If does not find identical match, performs a nearest match whereby it checks the coordinates: 
        1. to the left, 2. to the right 3. one down 4. one up
    Parameters: 
    ----------
    local_coordinate_array: numpy ndarray, channel x 2 
        0th array: x pos
        1st array y pos
    target_coordinate_array: numpy ndarray
        must be equal to or longer than local coordinates. 

    Return: 
    :list
        indices of target_coordinate array that correspond to local coordinates.
        If there was no match, the index returned will be target_coordinate_array+1.
        (a hack with which I will take care if giving these outputs a NaN)

    """
    local_x = local_coordinate_array[:,0]
    local_y = local_coordinate_array[:,1]
    target_x = target_coordinate_array[:,0]
    target_y = target_coordinate_array[:,1]
    nan_index = target_x.size 


    chan_idx = []
    for x,y in zip(local_x,local_y): 
        idx = np.where((target_x==x) & (target_y==y))[0]
        if idx.size==1: 
            chan_idx.append(idx[0])
        else: 
            idx = np.where((target_x==x-32) & (target_y==y))[0]
            if idx.size==1: 
                chan_idx.append(idx[0])
            else: 
                idx = np.where((target_x==x+32) & (target_y==y))[0]
                if idx.size==1: 
                    chan_idx.append(idx[0])
                else:
                    idx = np.where((target_x==x) & (target_y==y-15))[0]
                    if idx.size==1: 
                        chan_idx.append(idx[0])
                    else: 
                        idx = np.where((target_x==x) & (target_y==y+15))[0]
                        if idx.size==1: 
                            chan_idx.append(idx[0])
                        else: 
                            chan_idx.append(nan_index)
                            #print(x,y)

    return chan_idx   

def save_out_cluster_location(ibl_format_path,anatmap_paths=None):
    """
    function to save out anatomical location of clusters after the data has been alinged to the atlas using Mayo's tool

    options: if ther is channel_locations.json in path than matching happens using that file
    if there are alternative anatmap locations than the ibl_format files of those are used.

    todo: find suffucient anatmap nearest to the date of the file on the same implant

    Parameters: 
    -----------
    ibl_format_path: pathlib.Path
    anatmap_paths: list of pathlib.Path
        of the ibl_format folders of the anatmap 
    """
    # check if the channel_location.json exists
    matchable=False
    # if the current ibl format file is there, do the aligment with that
    if (ibl_format_path / 'channel_locations.json').is_file(): 
        chan_pos, allen_xyz, region_ID, region_acrnym = get_chan_coordinates(ibl_format_path)
        anatmap_paths = None
        matchable=True

    if anatmap_paths:
        chan_pos, allen_xyz, region_ID, region_acronym = zip(
        *[get_chan_coordinates(mypath) for mypath in anatmap_paths]
        )
        # concatenate the lists to arrays
        chan_pos,allen_xyz = np.concatenate(chan_pos),np.concatenate(allen_xyz)
        region_ID,region_acronym = np.concatenate(region_ID),np.concatenate(region_acronym) 
        
        # get the channels of interest and subselect
        channel_localCoordinates = np.load(ibl_format_path / 'channels.localCoordinates.npy')
        sel_idx = coordinate_matching(channel_localCoordinates,chan_pos)

        if np.max(sel_idx)==chan_pos.shape[0]: 
            check_which = (sel_idx==np.max(sel_idx))
            print('%.0f/%.0f clusters could not be assigned.' % (check_which.sum(),check_which.size))
            # concatenate a nan array to allen_xyz,acronyms and ID.
            allen_xyz = np.vstack((allen_xyz,np.ones(3)*np.nan))
            region_ID, region_acronym  = np.hstack((region_ID,np.nan)),np.hstack((region_acronym,np.nan))


        allen_xyz = allen_xyz[sel_idx]
        region_ID, region_acronym = region_ID[sel_idx], region_acronym[sel_idx]

        matchable = True

    # if the channels in ibl_format_path have been idenfied with some atlas correspondence
    # go ahead, and match the units and save 
    if matchable: 
        clus_channels = np.load(ibl_format_path / 'clusters.channels.npy')
        # get corresponding values for each cluster. 
        allen_xyz_clus = np.array([allen_xyz[clus_ch,:][:,np.newaxis] for clus_ch in clus_channels])	
        region_ID_clus = np.array([region_ID[clus_ch] for clus_ch in clus_channels])
        region_acronym_clus = np.array([region_acronym[clus_ch] for clus_ch in clus_channels])
        
        allencoords_ccf_apdvml = atlas.xyz2ccf(allen_xyz_clus[:,:,0]/1e6,ccf_order='apdvml') 
        allencoords_ccf_mlapdv = allencoords_ccf_apdvml[:,[2,0,1]]                  
    # save the output
        np.save(ibl_format_path / 'clusters.brainLocationIds_ccf_2017.npy',region_ID_clus)
        np.save(ibl_format_path / 'clusters.brainLocationAcronyms_ccf_2017.npy',region_acronym_clus)
        np.save(ibl_format_path / 'clusters.mlapdv.npy',allencoords_ccf_mlapdv)	
    else:
        print('we could not match channels with posititons.')  
