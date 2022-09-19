import json,re
import numpy as np

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
    chan_pos_x: list
        relative lateral channel position on neuropixel 
    chan_pos_y: list
        relative axial channel position on neuropixel
    allenx/y/z: list
        relative channel posititon on the allen atlas
    region_ID: list
        region ID that the channel is in. Can be used for hierarchical serarches in the atlas.
    region_acronym: list
        region name that the channel is in. 

    """
    channel_locations = open(root / 'channel_locations.json',)
    channel_locations = json.load(channel_locations)
    chan_names = np.array(list(channel_locations.keys()))
    chan_names = chan_names[['channel' in ch for ch in chan_names]] #get the channels only
    chan_IDs = [int(re.split('_',ch)[-1]) for ch in chan_names]
    allenx  = [channel_locations[ch]['x'] for ch in chan_names] 
    alleny  = [channel_locations[ch]['y'] for ch in chan_names] 
    allenz  = [channel_locations[ch]['z'] for ch in chan_names] 
    regionID = [channel_locations[ch]['brain_region_id'] for ch in chan_names] 
    region_acronym = [channel_locations[ch]['brain_region'] for ch in chan_names]
    chan_pos_x = [channel_locations[ch]['lateral'] for ch in chan_names] 
    chan_pos_y = [channel_locations[ch]['axial'] for ch in chan_names] 

    return chan_IDs,chan_pos_x,chan_pos_y,allenx,alleny,allenz,regionID,region_acronym

def read_chan_coord_npix2(root):
    """
    in the ibl_format the channels.localCoordinates is with the 1.0 spacing even for 2.0 probes. 
    This functrion takes care of this issue and reads the channel corrdinates correctly. 
    Parameters: 
    -----------
    root: Pathlib.path
        directory of ibl_format file 
    
    Returns: 
    --------
    numpy ndarray (chan x 2)
     
    """
    channel_localCoordinates = np.load(root / 'channels.localCoordinates.npy')
    channel_localCoordinates[:,0] = channel_localCoordinates[:,0]-11
    channel_localCoordinates[:,1] = channel_localCoordinates[:,1]*.75

    return channel_localCoordinates
