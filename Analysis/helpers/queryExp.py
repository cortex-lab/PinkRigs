import datetime
import pandas as pd
import re,inspect,json
import numpy as np
from pathlib import Path

def check_date_selection(date_selection,dateList):
    date_range = []
    date_range_called = False # when a from to type of date range called. Otherwise date_selection is treated as list of dates 
    if 'last' in date_selection:
        date_range_called = True 
        date_selection = date_selection.split('last')[1]
        date_range.append(datetime.datetime.today() - datetime.timedelta(days=int(date_selection)))
        date_range.append(datetime.datetime.today())
    else:
        if type(date_selection) is not list:
            # here the data selection becomes a list anyway
            date_range_called = True 
            date_selection=date_selection.split(':')
        
        for d in date_selection:
            date_range.append(datetime.datetime.strptime(d,'%Y-%m-%d'))   
        #if only one element
        if len(date_range) == 1:
            date_range.append(date_range[0])

    selected_dates = []
    for date in dateList:
        exp_date = datetime.datetime.strptime(date,'%Y-%m-%d')

        # check for the dates
        if date_range_called:
            IsGoodDate = (exp_date >= date_range[0]) & (exp_date <= date_range[1])
        else: 
            IsGoodDate= True in ([exp_date==date_range[i] for i in range(len(date_range))])           


        if IsGoodDate:
            selected_dates.append(True)
        else:
            selected_dates.append(False)
    return selected_dates

def queryCSV(subject='all',expDate='all',expDef='all',expNum = None):
    """ 
    python version to query experiments based on csvs produced on PinkRigs

    Parameters: 
    ----
    subject : str/list
        selected mice. Can be all, active, or specific subject names
    expDate : str/list
        selected dates. If str: Can be all,lastX,date range, or a single date
                        If list: string of selected dates (e.g. ['2022-03-15','2022-03-30'])
    expDef : str
        selected expdef or portion of the string of the the expdef name
    expNum: str/list 
        selected expNum
    
    Returns: 
    ----
    exp2checkList : pandas DataFrame 
        concatenated csv of requested experiments and its params 
    """


    root = Path(r'\\zinu.cortexlab.net\Subjects\PinkRigs')
    mainCSVLoc = root / '!MouseList.csv' 
    mouseList=pd.read_csv(mainCSVLoc)
    # look for selected mice
    if 'allActive' in subject:
        mouse2checkList = mouseList[mouseList.IsActive==1]['Subject']
    elif 'all' in subject: 
        mouse2checkList = mouseList.Subject
    else:
        if not isinstance(subject,list):
            subject = [subject]
        mouse2checkList = mouseList[mouseList.Subject.isin(subject)]['Subject']
    exp2checkList = []
    for mm in mouse2checkList:
        mouse_csv = root / (mm  + '.csv')
        if mouse_csv.is_file():
            expList = pd.read_csv(mouse_csv,dtype='str')
            expList.expDate=[date.replace('_','-').lower() for date in expList.expDate.values]
            if 'all' not in expDef:
                expList = expList[expList.expDef.str.contains(expDef)]
            if 'all' not in expDate: 
                # dealing with the call of posImplant based on the main csv. Otherwise one is able to use any date they would like 

                if 'postImplant' in expDate:
                    implant_date  = mouseList[mouseList.Subject == mm].P0_implantDate
                    # check whether mouse was implanted at all or not.
                    if ~implant_date.isnull().values[0]: 
                        implant_date = implant_date.values[0]
                        implant_date = implant_date.replace('_','-').lower()
                        implant_date_range = implant_date + ':' + expList.expDate.iloc[-1]
                        selected_dates = check_date_selection(implant_date_range,expList.expDate)
                    else: 
                        print('%s was not implanted. Why would you be calling it...?' % mm)
                        selected_dates = np.zeros(expList.expDate.size).astype('bool')

                else:  
                    selected_dates = check_date_selection(expDate,expList.expDate)

                expList = expList[selected_dates]
            if expNum:
                expNum = (np.array(expNum)).astype('str')
                _,idx,_ = np.intersect1d(expList.expNum.to_numpy(),expNum,return_indices=True)
                expList = expList.iloc[idx]  
            
            # add mouse name to list
            expList['Subject'] = mm

            exp2checkList.append(expList)

    exp2checkList = pd.concat(exp2checkList)
    # re-index
    exp2checkList = exp2checkList.reset_index(drop=True)
    

    return exp2checkList

class Bunch(dict):
    """ taken from iblutil
    A subclass of dictionary with an additional JavaSrcipt style dot syntax."""

    def __init__(self, *args, **kwargs):
        super(Bunch, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def copy(self):
        """Return a new Bunch instance which is a copy of the current Bunch instance."""
        return Bunch(super(Bunch, self).copy())

    def save(self, npz_file, compress=False):
        """
        Saves a npz file containing the arrays of the bunch.

        :param npz_file: output file
        :param compress: bool (False) use compression
        :return: None
        """
        if compress:
            np.savez_compressed(npz_file, **self)
        else:
            np.savez(npz_file, **self)

    @staticmethod
    def load(npz_file):
        """
        Loads a npz file containing the arrays of the bunch.

        :param npz_file: output file
        :return: Bunch
        """
        if not Path(npz_file).exists():
            raise FileNotFoundError(f"{npz_file}")
        return Bunch(np.load(npz_file))


def load_ONE_object(collection_folder,object,attributes='all'): 
    """
    function that loads any ONE object with npy extension
    ONE object = clollection_folder/object.attribute.expIDtag.extension 
    where extension is either .npy files or parquet table. 

    Parameters
    ----------
    collection_folder: pathlib.Path
    object: str 
        object name: e.g. spikes/clusters/_av_trials
    attributes: str/list
        if str: 'all': all attributes for the object 
        if list: list of strings with specified attributes 
    
    Returns: 
    ---------
    Bunch
        of  object.attribute

    """

    file_names = list(collection_folder.glob('%s.*' % object))
    object_names = [re.split(r"\.",file.name)[0] for file in file_names]
    attribute_names = [re.split(r"\.",file.name)[1] for file in file_names]
    extensions = [re.split(r"\.",file.name)[-1] for file in file_names]

    if 'all' in attributes: 
        attributes=attribute_names

    output = {}

    for f,o,a,e in zip(file_names,object_names,attribute_names,extensions):        
        # normally don't load any large data with this loader 
            if 'npy' in e: 
                if a in attributes: 
                    tempload = np.load(f)
                    if (tempload.ndim==2):
                        if (tempload.shape[1]==1): #if its the stupid matlab format, ravel
                            output[a] = tempload[:,0]
                        else:
                            output[a] = tempload        
                    else: 
                        output[a] = tempload

            elif 'pqt' in e: 
                if a in attributes:  # now I just not load the largeData
                    tempload = pd.read_parquet(f)
                    tempload = tempload.to_dict('list')

                    for k in tempload.keys():
                        output[k]=np.array(tempload[k])

            elif 'json' in e:
                if a in attributes:
                    tempload = open(f,)
                    tempload = json.load(tempload)
                    tempload = Path(tempload)
                    output[a] = tempload

    output = Bunch(output)

    return output




def load_data(data_name_dict=None,**kwargs):
    """
    Paramters: 
    -------------
    data_name_dict: str/dict
        if str: specific default dictionaries can be called, not implemented!
            'all'
            'ev_spk'
            'ev_cam_spk'

        if dict: nested dict that contains requested data
            {collection:{object:attribute}}
            
            note: 
            the raw ibl_format folder can also be called for spiking. 
            For this, one needs to give 'probe0_raw' or 'probe1_raw' as the collection namestring. 
            
    Returns: 
    -------------
    pd.DataFrame 
        collections  requested by data_name_dict are added as columns  to the original csvs.   

    Todo: implement cond loading,default params
        
    """
    recordings = queryCSV(**kwargs)

    if data_name_dict:

        collections = list(data_name_dict.keys())
        for collection in collections: 
            recordings[collection]=None
            for idx,rec in recordings.iterrows():
                # to do -- make it dependent on whether the extraction was done correctly 

                exp_folder = Path(rec.expFolder)
                if 'raw' in collection and 'probe' in collection:
                    probe_name = re.findall('probe\d',collection)[0]
                    probe_collection_folder = exp_folder / 'ONE_preproc' / probe_name
                    raw_path = list((probe_collection_folder).glob('_av_rawephys.path.*.json'))
                    if len(raw_path)==1:
                        # then a matching ephys file is found
                        ev_collection_folder = open(raw_path[0],)
                        ev_collection_folder = json.load(ev_collection_folder)
                        ev_collection_folder = Path(ev_collection_folder)
                    else: 
                        ev_collection_folder = probe_collection_folder                
                else:
                    ev_collection_folder = exp_folder / 'ONE_preproc' / collection
                
                objects = {}
                for object in data_name_dict[collection]:
                    objects[object] = load_ONE_object(ev_collection_folder,object,attributes=data_name_dict[collection][object])
                objects = Bunch(objects)
                recordings.loc[idx][collection] = objects

    return recordings


  

