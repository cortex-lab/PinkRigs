# general packages for managing paths of data
import os,glob,sys,json,re
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime as time # to sort only for a fixed amount of time
from pyhelpers import save_error_message

pd.options.mode.chained_assignment = None # disable warning, we will overwrite some rows when sortedTag changes 

# ibl ephys tools 
import spikeglx
from atlaselectrophysiology.extract_files import ks2_to_alf,extract_rmsmap,_sample2v
from ibllib.atlas import AllenAtlas
atlas = AllenAtlas(25) # always register to the 25 um atlas 

# PinkRig processing tools 
# import PinkRig ephys processng tools
pinkRig_path= glob.glob(r'C:\Users\*\Documents\Github\PinkRigs')
pinkRig_path = Path(pinkRig_path[0])
sys.path.insert(0, (pinkRig_path.__str__()))
from Admin.csv_pyhandlers import get_server_location 

from ReadSGLXData.readSGLX import readMeta

def check_date_selection(date_selection,date):
    import datetime 
    date_range = []

    if 'last' in date_selection: 
        date_selection = date_selection.split('last')[1]
        date_range.append(datetime.datetime.today() - datetime.timedelta(days=int(date_selection)))
        date_range.append(datetime.datetime.today())
    else:
        date_selection=date_selection.split(':')
        for d in date_selection:
            date_range.append(datetime.datetime.strptime(d,'%Y-%m-%d'))   

        if len(date_range) == 1:
            date_range.append(date_range[0])

    exp_date = datetime.datetime.strptime(date,'%Y-%m-%d')
    if (exp_date >= date_range[0]) & (exp_date <= date_range[1]):
        Out=True
    else:
        Out=False  

    return Out

def stage_queue(mouse_selection='',ks_folder='pyKS', date_selection='last3'):
    # the function will have a kwarg input structure where you can overwrite MasterMouseList with
    # which mice to sort -- FT or FT032
    # what dates to sort -- last10 from today or a range (2021-12-13:2021-12-20)
    # check
    
    print(mouse_selection)
    print(date_selection)

    # check which mice are active on Master csv
    root = get_server_location()
    master_csv = pd.read_csv(root / '!MouseList.csv')
    if mouse_selection=='allActive': 
        mice_to_check=master_csv[master_csv['IsActive']==1].Subject
    elif mouse_selection=='all': 
        mice_to_check=master_csv.Subject
    else: 
        mice_to_check = mouse_selection   

    new_recs_to_convert = []

    for mouse in mice_to_check:
        my_dates = pd.DataFrame()
        subject_csv_name = '%s.csv' % mouse
        subject_csv_path = root / subject_csv_name
        if subject_csv_path.is_file():
            subject_csv = pd.read_csv(root / subject_csv_name)
            my_dates = subject_csv.drop_duplicates('expDate')

            for my_path in my_dates.expFolder:
                mp = Path(my_path)

                server = mp.parts[0][:-1]
                subject= mp.parts[1]
                date = mp.parts[2]

                # only add the mice that need to be sorted if all criteria is fulfilled
                #if some dates have been subselected
                if check_date_selection(date_selection,date):
                    ephys_files = r'%s\%s\%s\ephys\**\*.ap.cbin' % (server,subject,date) 
                    ephys_files = glob.glob(ephys_files,recursive=True)

                    for ephys_file in ephys_files:
                        # look for pyKS folder with spike times in the same folder as ap.bin
                        KS_rez = r'%s\**\%s\**\spike_times.npy' % (os.path.dirname(ephys_file),ks_folder)
                        KS_rez = glob.glob(KS_rez,recursive=True) # should not be longer than 1?

                        # check if is there, and not empty
                        if not KS_rez:
                            # couldn't find the kilosort folder/rez file
                            KS_done = False
                        else:
                            if Path(KS_rez[0]).stat().st_size>0:
                                KS_done = True
                            else:
                                # file was 0kb
                                KS_done = False 

                        # also check for whether the ibl_format extraction is done
                        ibl_format = r'%s\**\%s\**\ibl_format\clusters.waveforms.npy' % (os.path.dirname(ephys_file),ks_folder)
                        ibl_format = glob.glob(ibl_format,recursive=True) # should not be longer than 1?

                        # check if is there, and not empty
                        if not ibl_format:
                            # couldn't find the kilosort folder/rez file
                            ibl_formatting_done = False
                        else:
                            if Path(ibl_format[0]).stat().st_size>0:
                                ibl_formatting_done = True
                            else:
                                # file was 0kb
                                ibl_formatting_done = False 


                        if  KS_done and not ibl_formatting_done:
                            ks_path = Path(ephys_file).parent / ks_folder

                            new_recs_to_convert.append([ks_path.__str__()])

    new_recs_to_convert = sum(new_recs_to_convert,[]) 
    print(new_recs_to_convert)
    # clean current queue
    queue_file = os.path.join(root,'Helpers','ibl_formatting_queue.csv')
    old_queue = pd.read_csv(queue_file,index_col=False)
    new_queue = old_queue[old_queue['doneTag'] != 1]

    isnew = len(new_recs_to_convert)>0
    # if there are new recs to sort, then overwrite queue
    if isnew:
        added_recs = pd.DataFrame(new_recs_to_convert,columns=[old_queue.columns[0]])
        added_recs[old_queue.columns[1]]=0
        new_queue = pd.concat([new_queue,added_recs])
        # remove duplicates
        new_queue = new_queue.drop_duplicates('ksFolderPath')

    new_queue.to_csv(queue_file,index = False)
    print('%d files are waiting to be converted ...'
        % (len(new_queue[new_queue['doneTag']==0])))

def extract_data_PinkRigs(ks_path, ephys_path, out_path,do_raw_files=False):
    efiles = spikeglx.glob_ephys_files(ephys_path)

    for efile in efiles:
        if efile.get('ap') and efile.ap.exists():
            ks2_to_alf(ks_path, ephys_path, out_path, bin_file=efile.ap,
                       ampfactor=_sample2v(efile.ap), label=None, force=True)

            # I might need to rewrite the channels.localCoordinate files. These seem to be wrong
            # on npix 2.0. (as in: incorrect spcing and positional identifier.)
            if do_raw_files:
                extract_rmsmap(efile.ap, out_folder=out_path, spectra=False)
        if efile.get('lf') and efile.lf.exists() and do_raw_files:
            extract_rmsmap(efile.lf, out_folder=out_path)

def ks_to_ibl_format(ephys_path,ks_folder='pyKS',recompute=False):
    # Path to KS output
    if 'pyKS' in ks_folder: 
        ks_folder = r'pyKS\output'
        
    ks_path = ephys_path / ks_folder
    # Save path
    out_path = ks_path / 'ibl_format'
    out_path.mkdir(parents=False, exist_ok=True) # make directory if it does not exist

    # extract the data to ibl_format if it has not been done already.
    if not (out_path / 'cluster_matrics.tsv').is_file() or recompute:
        print('converting data to IBL format ...')
        extract_data_PinkRigs(ks_path, ephys_path, out_path,do_raw_files=False) 

    return out_path

    
def run_batch_ibl_formatting(run_for=2):
    # get queue
    root = get_server_location() / 'Helpers'
    queue_csv_file = root / 'ibl_formatting_queue.csv' 
    queue_csv = pd.read_csv(queue_csv_file)
    print('checking the queue...')

    start_time = time.now()
    start_hour = start_time.hour+start_time.minute/60

    print('current hour is %.2f' % start_hour)

    print('starting my work on queue..')
    for idx,rec in queue_csv.iloc[::-1].iterrows():
        #check if recording is not being sorted already 
        if rec.doneTag==0: 
            check_time = time.now()
            check_hour = check_time.hour+check_time.minute/60
            if check_hour<(start_hour+run_for): 
                print('still within my time limit... ')
                try:
                    input_dir = Path(rec.ksFolderPath)     
                    print(input_dir)  
                    output_dir = ks_to_ibl_format(input_dir.parent,ks_folder=input_dir.name,recompute=False)     

                    
                    queue_csv.doneTag.iloc[idx]= 1
                    queue_csv.to_csv(queue_csv_file,index = False)
                except: 
                    # save error message 
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    save_error_message(output_dir / 'ibl_formatting_error.json',err_type=exc_type,err_message=exc_obj,err_traceback=exc_tb)

                    # update csv                
                    queue_csv.doneTag.iloc[idx]= -1
                    queue_csv.to_csv(queue_csv_file,index = False)  


def add_anat_to_ibl_format(ephys_path,ks_folder='pyKS',recompute=True):
    # Path to KS output
    if 'pyKS' in ks_folder: 
        ks_folder = r'pyKS\output'
        
    ks_path = ephys_path / ks_folder
    # Save path
    out_path = ks_path / 'ibl_format'
    out_path.mkdir(parents=False, exist_ok=True) # make directory if it does not exist

    # extract the data to ibl_format if it has not been done already.
    if not (out_path / 'cluster_matrics.tsv').is_file() or recompute:
        print('converting data to IBL format ...')
        extract_data_PinkRigs(ks_path, ephys_path, out_path,do_raw_files=True) 

    # get the probe tracks if they exist.
    subject_path = ephys_path.parents[3]
    brainreg_path = subject_path / r'histology\registration\brainreg_output\manual_segmentation\standard_space\tracks'
    if brainreg_path.is_dir():
        # get some information from meta file 
        meta = readMeta(list(ephys_path.glob('*.ap.cbin'))[0])

        # serial number
        probe_sn = meta['imDatPrb_sn']

        # shanks that have been recorded. 
        imro = meta['snsShankMap']
        channel_data = re.findall(r"\d+:\d+:\d+:\d+",imro)
        shank_idx = np.array([np.array(re.split(':',chandat)).astype('int')[0] for chandat in channel_data])
        recorded_shanks = np.unique(shank_idx).astype('str')
            # get the xyz_picks if the histology exist.

        for shank in recorded_shanks:
            shank_file_name = 'SN%s_shank%s.npy' % (probe_sn,shank)
            shank_anat_path = brainreg_path / shank_file_name

            if shank_anat_path.is_file():
                # Load in coordinates of track in CCF space (order - apdvml, origin - top, left, front voxel
                xyz_apdvml = np.load(shank_anat_path)
                # Convert to IBL space (order - mlapdv, origin - bregma)
                xyz_mlapdv = atlas.ccf2xyz(xyz_apdvml, ccf_order='apdvml') * 1e6
                xyz_picks = {'xyz_picks': xyz_mlapdv.tolist()}

                # save file the ibl_format output path - the format is following the curernt requirements of the GUI, i.e. 
                # xyz_picks.json if single shank recording
                # else xyz_picks_shank0.json 
                if recorded_shanks.size==1:
                    with open(Path(out_path, 'xyz_picks.json'), "w") as f:
                        json.dump(xyz_picks, f, indent=2)
                else:
                    with open(Path(out_path, 'xyz_picks_shank%s.json' % shank), "w") as f:
                        json.dump(xyz_picks, f, indent=2)
            else: 
                print('the probe/shank IDs do not match.')                
    else: 
        print('the histology does not seem to have been done. ')


if __name__ == "__main__":
   stage_queue(mouse_selection=sys.argv[1],ks_folder = sys.argv[2],date_selection=sys.argv[3])
   #stage_queue(mouse_selection='allActive',ks_folder = 'pyKS', date_selection='last7')
   run_batch_ibl_formatting(run_for=2)


