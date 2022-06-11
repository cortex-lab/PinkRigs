import numpy as np
from mtscomp import compress
import glob
import re
import os

localDataFolder = 'D:\ephysData\**\**\*ap.bin'
locFiles = glob.glob(localDataFolder)

for locFile in locFiles:
    # Compress a .bin file into a pair .cbin (compressed binary file) and .ch (JSON file).
    cbinFile = re.sub('ap.bin','ap.cbin',locFile)
    chFile = re.sub('ap.bin','ap.ch',locFile)
    if not os.path.exists(cbinFile):
        compress(locFile, cbinFile, chFile, sample_rate=30000., n_channels=385, 
                 dtype=np.int16, check_after_compress=True)
        
        # os.remove(locFile)
        
