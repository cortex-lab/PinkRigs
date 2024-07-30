from pathlib import Path
import pandas as pd
import numpy as np
from skimage import io

from Admin.csv_queryExp import queryCSV,load_data

#### these are now going to be historic .....
def select_opto_sessions(recdat): 
    print('loading opto metadata ...')
    laser_powers = []
    stim_hemisphere = []
    for _,rec in recdat.iterrows(): 
        check_opto_meta = list((Path(rec.expFolder)).glob('*optoMetaData.csv'))
        if len(check_opto_meta):
            # read the csv and data to recdat
            optoMeta = pd.read_csv(check_opto_meta[0])
            laser_powers.append(optoMeta.LaserPower_mW.values[0])
            stim_hemisphere.append(optoMeta.Hemisphere.values[0])
        else: 
            laser_powers.append(0)
            stim_hemisphere.append(np.nan) 

    recdat = recdat.assign(
        laser_power = laser_powers,
        stimulated_hemisphere = stim_hemisphere
    ) 
    
    return recdat

def query_opto(power=30,hemi='L',data_dict = None, **kwargs):
    """
    function that imports the opto metadata and allows to load the rest of the data on those sessions 
    
    """

    recordings = queryCSV(**kwargs)
    recordings = select_opto_sessions(recordings)
    recordings = recordings[(recordings.laser_power==power) & (recordings.stimulated_hemisphere == hemi)]
    optodata_list  = [load_data(subject = rec.Subject,expDate = rec.expDate,expNum = rec.expNum,data_name_dict=data_dict)for _,rec in recordings.iterrows()]
    optodata_list = pd.concat(optodata_list)

    return optodata_list


def get_relative_eYFP_intensity(track_path,sq=1):
    """
    function to calculate the intenstity of the YFP on the green channel 

    method: 
    1) take the sum intensity below the tip of the cannula in a box sized sq mm
    2) Min-max to the autofluorescence, calculated 2-4 boxes below 

    Parameters:
    -----------
    track_path: pathlib.Path to the track in manual segmentation via brainreg
    sq: size of the square we calculate in, in mm

    Return: float
        relative intensity

    """

    # load the positition of the tip based on the track
    tip = np.load(track_path)[0] # in ap,dv,ml

    tip_ = (np.floor(tip/25)).astype('int') # in voxel idx, ap,dv,ml -same as image indices 
    
    # get square size in voxel
    sq_ = int(sq*1000/25) 

    # load the image toff relative to the track
    gsource=track_path.parents[3] / 'downsampled_standard_green.tiff'
    g=io.imread(gsource)

    # get the image at the top of the cannula
    # the top of the cannula should be circa 13 voxel above the labelled tip
    tip_im = g[tip_[0],tip_[1]-10:tip_[1]+sq_,(tip_[2]-int(sq_/2)):(tip_[2]+int(sq_/2))]


    autofluor_averages = [np.mean(
        g[tip_[0],tip_[1]+sq_*(i+2)-10:tip_[1]+sq_*(i+2),tip_[2]:tip_[2]+sq_]) for i in range(3)]

    rel_intensity = (np.mean(tip_im)-np.mean(autofluor_averages))/(np.mean(tip_im)+np.mean(autofluor_averages))
    

    return rel_intensity