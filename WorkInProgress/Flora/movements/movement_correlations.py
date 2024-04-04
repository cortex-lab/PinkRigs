import sys
import numpy as np 
import matplotlib.pyplot as plt 

sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from loaders import load_for_movement_correlation
recordings = load_for_movement_correlation(dataset='postactiveWithSpikes',recompute_data_selection=True)