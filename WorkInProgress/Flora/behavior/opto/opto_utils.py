from pathlib import Path
import pandas as pd
import numpy as np

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

    