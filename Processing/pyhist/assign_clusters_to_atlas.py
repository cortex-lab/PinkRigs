import json,re,glob,sys,datetime
import numpy as np
import pandas as pd
from pathlib import Path
from helpers.atlas import AllenAtlas
from shutil import copyfile
atlas = AllenAtlas(25)

# PinkRig specific imports 
pinkRig_path= glob.glob(r'C:\Users\*\Documents\Github\PinkRigs')
pinkRig_path = Path(pinkRig_path[0])
sys.path.insert(0, (pinkRig_path.__str__()))
from Processing.pykilo.ReadSGLXData.readSGLX import readMeta
from Admin.csv_queryExp import load_data,get_recorded_channel_position

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



    return chan_pos,allencoords_xyz,np.array(regionID),np.array(region_acronym)


def coordinate_matching(local_coordinate_array,target_coordinate_array):
    """
    performs a coordinate matching bsed on xy positions using the channels.localCoordinates.npy output of the ibl format  
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

def save_to_common_anatmap(one_path,probe,shank,botrow,date):
    ibl_format_path = open(list(one_path.glob('*.path.*.json'))[0],)
    ibl_format_path = Path(json.load(ibl_format_path))        
    chanfile = (ibl_format_path / 'channel_locations.json')
    if chanfile.is_file(): 
        #redump to parent histology
        output_folder = ibl_format_path.parents[6] / ('histology/registered_anatmaps/%s' % date)            
        output_folder.mkdir(parents=True,exist_ok=True)         
        stub = 'channel_locations_%s_shank%.0d_botrow%.0d.json' % (probe,shank,botrow)
        copyfile(chanfile,(output_folder / stub))

def save_out_cluster_location(one_path,anatmap_paths=None):
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
    # ibl_format_path

    ibl_format_path = open(list(one_path.glob('*.path.*.json'))[0],)
    ibl_format_path = Path(json.load(ibl_format_path))
        
    # check if the channel_location.json exists
    matchable=False
    # if the current ibl format file is there, do the aligment with that  
    
    if (ibl_format_path / 'channel_locations.json').is_file(): 
        chan_pos, allen_xyz, region_ID, region_acronym = get_chan_coordinates(ibl_format_path)
        anatmap_paths = None
        matchable=True

    if anatmap_paths:
        # here get the nearest sparseNoise recordings
        chan_pos, allen_xyz, region_ID, region_acronym = zip(
        *[get_chan_coordinates(mypath) for mypath in anatmap_paths]
        )
        # concatenate the lists to arrays
        chan_pos,allen_xyz = np.concatenate(chan_pos),np.concatenate(allen_xyz)
        region_ID,region_acronym = np.concatenate(region_ID),np.concatenate(region_acronym) 

        # get the channels of interest and subselect
        channel_localCoordinates = np.load(ibl_format_path / 'channels.localCoordinates.npy')

        if channel_localCoordinates.shape[0]==384:
            pass
        else:
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
        a = list(one_path.glob('*path.*.json'))[0]
        stub = re.split('[.]',a.__str__())[-2]
        clus_channels = np.load(one_path / ('clusters.channels.%s.npy' % stub))
        # get corresponding values for each cluster. 
        allen_xyz_clus = np.array([allen_xyz[clus_ch,:][:,np.newaxis] for clus_ch in clus_channels])	
        region_ID_clus = np.array([region_ID[clus_ch] for clus_ch in clus_channels])
        region_acronym_clus = np.array([region_acronym[clus_ch] for clus_ch in clus_channels])
        
        # some units can be in "void" (I imagine mostly noise)
        # those locations will error for tte converions 
        allen_xyz_clus = allen_xyz_clus[:,:,0]
        allen_xyz_clus[region_ID_clus==0] = np.nan

        allencoords_ccf_apdvml = atlas.xyz2ccf(allen_xyz_clus/1e6,ccf_order='apdvml') 
        allencoords_ccf_mlapdv = allencoords_ccf_apdvml[:,0,[2,0,1]]                  
    # save the output
        np.save(one_path / ('clusters.brainLocationIds_ccf_2017.%s.npy' % stub),region_ID_clus)
        np.save(one_path / ('clusters.brainLocationAcronyms_ccf_2017.%s.npy' % stub),region_acronym_clus)
        np.save(one_path / ('clusters.mlapdv.%s.npy' % stub),allencoords_ccf_mlapdv)	
    else:
        print('we could not match channels with posititons for %s' % one_path.__str__())  

def read_probeSN_from_folder(folderpath):
    """
    read meta file from parent folder the .ap.bin file is in 
    Parameters: 
    -----------
    folderpath: pathlib.Path

    Returns: 
    --------
        : str

    """
    meta = readMeta(list((folderpath).glob('*.ap.cbin'))[0])
    probe_sn = meta['imDatPrb_sn']

    return probe_sn 

def read_probeSN_from_one_folder(one_path):
    """
    read the probe serial number from the one path
    Parameters: 
    -------------
    one_path: pathlib.Path

    Returns: 
    ---------
        :str

    """

    a = list(one_path.glob('*path.*.json'))[0]
    a = a.__str__()
    probeSN = a[-16:-5]
    return probeSN

def get_anatmap_path_same_day(one_path):
    """
    function to get ibl_format_paths that already contain the channel_locations.json files and match the probe serial number of the input puath
    Parameters: 
    -----------
    ibl_format_path: pathlib.Path

    Returns: 
    --------
        :list[pathlib.Path]
    """

    anatmap_list = list(one_path.parents[2].glob("**/kilosort2/ibl_format/channel_locations.json")) 
    anatmap_paths = [p.parent for p in anatmap_list]
    # check if the serial number is matching
    target_SN = read_probeSN_from_one_folder(one_path)
    is_SN_match = [read_probeSN_from_folder(p.parents[1])==target_SN for p in anatmap_paths]
    anatmap_paths = (np.array(anatmap_paths)[is_SN_match]).tolist()

    return anatmap_paths

def call_for_anatmap_recordings(subject='AV025',probe='probe0',near_date=None,depth_selection = 'auto'): 
    """
    function to call which recordings should be used for anatomy
    basically this function searches for single shank recordings
    and gets unique depths for each shanks

    Paramters:
    ---------
    subject: str
        subject name 
    probe: str
        'probe0' or 'probe1'
    near_date: None/str
        a string of a date. in this case the algorithm will look for the nearest date before date given by near_date
    depth selection: str
        specific modes of depth selection for single shanks. Options:
        auto - searches exclusively for botrow 0 and 192 single shanks. 

    """
    data_dict = {
    ('%s_raw' % probe):{'channels':'all'}
    }
    
    sn_recs=load_data(data_name_dict=data_dict,subject=subject,expDef='sparseNoise')
    spont_recs=load_data(data_name_dict=data_dict,subject=subject,expDef='spontaneous')
    recdat = pd.concat((sn_recs,spont_recs))


    shank_range,depth_range = zip(*[get_recorded_channel_position(rec[('%s_raw' % probe)].channels) for _,rec in recdat.iterrows()])
    # should also check whether the brain region id really exists. 

    recdat = recdat.assign(
        shank_range = shank_range, 
        depth_range = depth_range
    )

    recdat = recdat.dropna(subset=['shank_range','depth_range'])

    is_single_shank = [(rec.shank_range[1] - rec.shank_range[0])<35 for _,rec in recdat.iterrows()]
    recdat = recdat[is_single_shank]
    recdat = recdat.assign(
        shank = [int(sh[0]/200) for sh in recdat.shank_range],
        botrow = [int(site[0]/15) for site in recdat.depth_range]
    )



    # and contain all unique depths for each shank

    out_dat = pd.DataFrame(columns=recdat.columns)
    shanks = np.unique(recdat.shank)
    for sh in shanks: 
        recdat_shank = recdat[recdat.shank==sh]

        if 'auto' in depth_selection:
            unique_depths = [0,192]
        else:    
            unique_depths = np.unique(recdat_shank.botrow)


        for my_d in unique_depths:
            recdat_shank_d = (recdat_shank[recdat_shank.botrow==my_d]).copy()
            # and select either the nerest to an asked date all the 1st post Implant
            all_dates = [datetime.datetime.strptime(d,'%Y-%m-%d') for d in recdat_shank_d.expDate]
            if near_date:
                selected_date = datetime.datetime.strptime(near_date,'%Y-%m-%d')
                possible_dates_prior = [d  for d in all_dates if d<=selected_date]
                date_for_shank = max(possible_dates_prior)
            else:
                date_for_shank = min(all_dates)
            
            date_for_shank = date_for_shank.strftime('%Y-%m-%d')
            selected_rec = recdat_shank_d[recdat_shank_d.expDate==date_for_shank]
            out_dat = pd.concat([out_dat,selected_rec])
    
    out_dat = out_dat.drop_duplicates(subset=['shank_range','depth_range'])

    return out_dat