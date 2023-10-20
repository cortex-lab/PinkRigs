
import sys
from pathlib import Path
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from Admin.csv_queryExp import load_ephys_independent_probes,simplify_recdat
from Analysis.neural.utils.spike_dat import bincount2D

from utils import save_pickle

basepath = Path(r'C:\Users\Flora\Documents\Processed data\rastermap')

mname = 'AV030'
expDate = '2022-12-11'
probe = 'probe0'
sess='multiSpaceWorld'

session = { 
    'subject':mname,
    'expDate': expDate,
    'expDef': sess,
    'probe': probe
}

ephys_dict =  {'spikes': ['times', 'clusters'],'clusters':'all'}
other_ = {'events': {'_av_trials': 'table'}}
recordings = load_ephys_independent_probes(ephys_dict=ephys_dict,add_dict=other_,**session)
ev,spikes,clusters,_,_ = simplify_recdat(recordings.iloc[0],probe='probe')

