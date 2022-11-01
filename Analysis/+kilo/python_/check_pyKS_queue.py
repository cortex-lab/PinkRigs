import os,glob
from pathlib import Path
import pandas as pd
import numpy as np
import sys

# get PinkRig handlers 
pinkRig_path= glob.glob(r'C:\Users\*\Documents\Github\PinkRigs')
pinkRig_path = Path(pinkRig_path[0])
sys.path.insert(0, (pinkRig_path.__str__()))
from Admin.csv_pyhandlers import get_server_location 
import datetime 

def check_date_selection(date_selection,date):
    
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

def stage_KS_queue(mouse_selection='',date_selection='last3',resort = False):
    # the function will have a kwarg input structure where you can overwrite MasterMouseList with
    # which mice to sort -- FT or FT032
    # what dates to sort -- last10 from today or a range (2021-12-13:2021-12-20)
    # check
    
    print('mice selected: %s' % mouse_selection)
    print('dates selected: %s' % date_selection)

    # check which mice are active on Master csv
    root = get_server_location()

    master_csv = pd.read_csv(root / '!MouseList.csv')
    if mouse_selection=='allActive': 
        mice_to_check=master_csv[master_csv['IsActive']==1].Subject
    elif mouse_selection=='all': 
        mice_to_check=master_csv.Subject
    else: 
        mice_to_check = mouse_selection   

    new_recs_to_sort = []

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
                # that is: 

                    #if some dates have been subselected
                if check_date_selection(date_selection,date):
                    ephys_files = r'%s\%s\%s\ephys\**\*.ap.cbin' % (server,subject,date) 
                    ephys_files = glob.glob(ephys_files,recursive=True)

                    for ephys_file in ephys_files:


                        # look for pyKS folder with spike times in the same folder as ap.bin
                        KS_rez = r'%s\**\pyKS\**\spike_times.npy' % (os.path.dirname(ephys_file))
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

                        # override KS_done if resorting is requested 
                        if resort: 
                            KS_done = False

                        # override KS_done if the file was modified in the last hour. 
                        # check when the ephys file was created and don't sort if it's less than an hour. 
                        last_modification_time = Path(ephys_file).stat().st_mtime
                        modification_thr = 1 # 1 hr
                        is_recently_modified_file = ((datetime.datetime.now().timestamp()-last_modification_time)/3600)<modification_thr
                        if is_recently_modified_file:
                            KS_done = True

                        if not KS_done:
                            print(ephys_file)
                            new_recs_to_sort.append(glob.glob(ephys_file,recursive=True))

    new_recs_to_sort = sum(new_recs_to_sort,[]) 
    print(new_recs_to_sort)
    # clean current queue
    queue_file = root/ r'Helpers/pykilosort_queue.csv'
    old_queue = pd.read_csv(queue_file,index_col=False)
    new_queue = old_queue[old_queue['sortedTag'] != 1]

    isnew = len(new_recs_to_sort)>0
    # if there are new recs to sort, then overwrite queue
    if isnew:
        added_recs = pd.DataFrame(new_recs_to_sort,columns=[old_queue.columns[0]])
        added_recs[old_queue.columns[1]]=0
        new_queue = pd.concat([new_queue,added_recs])
        # remove duplicates
        new_queue = new_queue.drop_duplicates('ephysName')

    new_queue.to_csv(queue_file,index = False)
    print('%d files are waiting to be sorted ...'
        % (len(new_queue[new_queue['sortedTag']==0])))

if __name__ == "__main__":
   stage_KS_queue(mouse_selection=sys.argv[1],date_selection=sys.argv[2])
   #stage_KS_queue(mouse_selection='allActive',date_selection='last10')