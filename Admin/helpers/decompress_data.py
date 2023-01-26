# -*- coding: utf-8 -*-
"""
Created on Tue May 31 22:57:07 2022

@author: Célian
"""
from mtscomp import Reader
import re
import os
import sys

def mainDecompress(cbinFile='', chFile=''):
    # Define a reader to decompress a compressed array.
    r = Reader()
    # Open the compressed dataset.
    binFile = re.sub('ap.cbin','ap.bin',cbinFile);
    r.open(cbinFile, chFile)
    # We can decompress into a new raw binary file on disk.
    r.tofile(binFile)
    r.close()

    os.remove(cbinFile)
    os.remove(chFile)

if __name__ == "__main__":
   mainDecompress(cbinFile=sys.argv[1],chFile=sys.argv[2])