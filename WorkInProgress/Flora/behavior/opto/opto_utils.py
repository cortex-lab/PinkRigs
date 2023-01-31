from pathlib import Path
import pandas as pd
import numpy as np

from Admin.csv_queryExp import queryCSV,load_data

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
