import os,glob
from pathlib import Path
import pandas as pd
import numpy as np

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

def stage_KS_queue(mouse_selection='',date_selection='last3'):
    # the fuction will have a kwarg input structure where you can overwrite MasterMouseList with
    # which mice to sort -- FT or FT032
    # what dates to sort -- last10 from today or a range (2021-12-13:2021-12-20)
    # check 

    # check which mice are active on Master csv
    root = r'\\zserver.cortexlab.net\Code\AVrig'
    master_csv = pd.read_csv(os.path.join(root,'!MasterMouseList.csv'))
    mice_to_check=master_csv[master_csv['IsActive']==1].Subject

    new_recs_to_sort=[]

    for i,mouse in enumerate(mice_to_check):
        my_dates = pd.DataFrame()
        subject_csv = pd.read_csv(os.path.join(root,'%s.csv' % mouse))
        my_dates = subject_csv[subject_csv.ephys>0].drop_duplicates('expDate')

        for i,my_path in enumerate(my_dates.path):
            mp= Path(my_path)

            server = mp.parts[0][:-1]
            subject= mp.parts[1]
            date = mp.parts[2]



            # only add the mice that need to be sorted if all criteria is fulfilled
            # that is: 
            # if the mouse names are subselected 
            if mouse_selection in subject: 
                #if some dates have been subselected
                if check_date_selection(date_selection,date):
                    # then check kilosort 
                    #potential kilosort folder
                    KS_folders = r'\%s\%s\%s\ephys\**\kilosort\**\rez2.mat' % (server,subject,date)
                    KS_folders = glob.glob(KS_folders, recursive=True)

                    # check if KS was sorted correctly previously
                    KS_started = len(KS_folders)>0
                    if KS_started:
                        for k in KS_folders:                    
                            KS_done = Path(k).stat().st_size>0
                    else:
                        KS_done=KS_started

                    # check if even if it wasn't completed, it might have errored and cannot be sorted
                    

                    # add to queue if not    
                    if not KS_done:                  
                        # get the ap file that ought to be sorted 
                        ephys_files = r'%s\%s\%s\ephys\**\**\*.ap.bin' % (server,subject,date)    
                        new_recs_to_sort.append(glob.glob(ephys_files,recursive=True))




    new_recs_to_sort = sum(new_recs_to_sort,[])


    isnew = len(new_recs_to_sort)>0
    # if there are new recs to sort, then overwrite queue
    if isnew:
        
        queue_file = os.path.join(root,'kilosort_queue.csv')
        old_queue = pd.read_csv(queue_file,index_col=False)

    #     #backup the old file
    #     now = datetime.datetime.now()
    #     backup_path = os.path.join(root,'Backups\%s_kilosort_queue.csv') % (now.strftime('%Y-%m-%d'))
    #     old_queue.to_csv(backup_path)

        added_recs = pd.DataFrame(new_recs_to_sort,columns=[old_queue.columns[0]])
        added_recs[old_queue.columns[1]]=0
        new_queue = pd.concat([old_queue,added_recs])
        # remove what has already been queing
        new_queue=new_queue.drop_duplicates('ephysName')

        new_queue.to_csv(queue_file,index = False)
    
        print('%d files are waiting to be sorted ...'
              % (len(new_queue[new_queue['sortedTag']==0])))
