import numpy as np
from mtscomp import compress
import re
import os
import sys

def mainCompress(binFile=''):
    # Compress a .bin file into a pair .cbin (compressed binary file) and .ch (JSON file).
    cbinFile = re.sub('ap.bin','ap.cbin',binFile)
    chFile = re.sub('ap.bin','ap.ch',binFile)

    if not os.path.exists(cbinFile) and not os.path.exists(chFile):
        compress(binFile, cbinFile, chFile, sample_rate=30000., n_channels=385, 
                 dtype=np.int16, check_after_compress=True)

if __name__ == "__main__":
   mainCompress(binFile=sys.argv[1])
