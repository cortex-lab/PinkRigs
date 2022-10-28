import numpy as np
from mtscomp import compress
import glob
import re
import os
import datetime

localDataFolder = 'D:\ephysData\**\**\*ap.bin'
locFiles = glob.glob(localDataFolder)



for locFile in locFiles:
    # Compress a .bin file into a pair .cbin (compressed binary file) and .ch (JSON file).
    cbinFile = re.sub('ap.bin','ap.cbin',locFile)
    chFile = re.sub('ap.bin','ap.ch',locFile)

    rec_timestamp = os.path.getmtime(locFile)
    rec_datestamp = datetime.datetime.fromtimestamp(timestamp)
    today_datestamp = datetime.datetime.today()
    rec_age = today_datestamp - rec_datestamp
    rec_age_min = rec_age.days*24*60+rec_age.seconds/60

    if not os.path.exists(cbinFile) & (rec_age_min > 60):
        compress(locFile, cbinFile, chFile, sample_rate=30000., n_channels=385, 
                 dtype=np.int16, check_after_compress=True)
        
        os.remove(locFile)
        
