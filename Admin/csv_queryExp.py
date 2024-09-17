
"""
a bunch of functions that unify csv handling across 
python codes, planned to be somewhat equivalent to +csv. funcs in the matlab base 

"""
import datetime
import glob
import json
import re
import sys
from itertools import compress
from pathlib import Path

import numpy as np
import pandas as pd

# get PinkRig handlers
pinkRig_path = glob.glob(r'C:\Users\*\Documents\Github\PinkRigs')

if len(pinkRig_path) > 0:
    pinkRig_path = Path(pinkRig_path[0])
    sys.path.insert(0, (pinkRig_path.__str__()))


def get_csv_location(which):
    """
    func equivalent to getLocation in matlab

    """
    server = Path(r'\\znas.cortexlab.net\Code\PinkRigs')
    if 'main' in which:
        SHEET_ID = '1_hcuWF68PaUFeDBvXAJE0eUmyD9WSQlWtuEqdEuQcVg'
        SHEET_NAME = 'Sheet1'
        csvpath = f'https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={SHEET_NAME}'
    elif 'ibl_queue' in which:
        csvpath = server / r'Helpers/ibl_formatting_queue.csv'
    elif 'pyKS_queue' in which:
        csvpath = server / r'Helpers/pykilosort_queue.csv'
    elif 'training_email' in which:
        csvpath = server / r'Helpers\AVrigEmail.txt'
    else:
        csvpath = server / ('SubjectCSVs/%s.csv' % which)

    return csvpath


def get_server_list():
    """get the list of servers we are currently searching on

    Returns:
        list: pathlib.Path of servers
    """
    server_list = [
        Path(r'\\zinu.cortexlab.net\Subjects'),
        Path(r'\\zaru.cortexlab.net\Subjects'),
        Path(r'\\znas.cortexlab.net\Subjects'),
        Path(r'\\zortex.cortexlab.net\Subjects')
    ]

    return server_list


def check_date_selection(date_selection, date_list):
    """
   funct to match a called date range to a list of dates (indicating all epxeriments in the csv, for example)

    Parameters:
    -----------
    date selection: If str: Can be all,lastX,date range, or a single date
                    If list: string of selected dates (e.g. ['2022-03-15','2022-03-30'])
        corresponds to the selected dates
         
    date_list: list
        corresponds to all dates to match to 
    
    Return: list
        list of dates selected from dateList that pass the criteria determined by date_selection
    """
    date_range = []
    date_range_called = False  # when a from to type of date range called. Otherwise date_selection is treated as list of dates
    if 'previous' in date_selection:
        date_range_called = True
        date_selection = date_selection.split('previous')[1]
        date_range.append(datetime.datetime.today() - datetime.timedelta(days=int(date_selection)))
        date_range.append(datetime.datetime.today())
    else:
        if type(date_selection) is not list:
            # here the data selection becomes a list anyway
            date_range_called = True
            date_selection = date_selection.split(':')

        for d in date_selection:
            date_range.append(datetime.datetime.strptime(d, '%Y-%m-%d'))
            # if only one element
        if len(date_range) == 1:
            date_range.append(date_range[0])

    selected_dates = []
    for date in date_list:
        exp_date = datetime.datetime.strptime(date, '%Y-%m-%d')

        # check for the dates
        if date_range_called:
            IsGoodDate = (exp_date >= date_range[0]) & (exp_date <= date_range[1])
        else:
            IsGoodDate = True in ([exp_date == date_range[i] for i in range(len(date_range))])

        if IsGoodDate:
            selected_dates.append(True)
        else:
            selected_dates.append(False)
    return selected_dates


def bombcell_sort_units(clusdat, max_peaks=2, max_throughs=1,
                        is_somatic=1, min_spatial_decay_slope=-0.003,
                        min_waveform_duration=100, max_waveform_duration=800,
                        max_waveform_baseline_fraction=.3, max_percentage_spikes_missing=20,
                        min_spike_num=300, max_refractory_period_violations=.1, min_amp=20, minSNR=.1, max_drift=500,
                        min_presence_ratio=.2):
    """
    classifier that sorts units into good,mua and noise, based on bombcell parameters

    Parameters: 
    ----------
    see bombcell documentataion 


    Returns:
    -------
        	: np. array 
        bombcell class (mua/good/noise)
    """

    # any unit that is not discarded as noise or selected as well isolated is mua.
    # maybe there ought to be an option to classify well isolated axonal units...

    bombcell_class = np.empty(clusdat.nPeaks.size, dtype="object")
    bombcell_class[:] = 'mua'
    # assign noise 

    ix = (
            (clusdat.nPeaks > max_peaks) |
            (clusdat.nTroughs > max_throughs) |
            (clusdat.isSomatic != is_somatic) |
            (clusdat.spatialDecaySlope > min_spatial_decay_slope) |
            (clusdat.waveformDuration_peakTrough < min_waveform_duration) |
            (clusdat.waveformDuration_peakTrough > max_waveform_duration) |
            (clusdat.waveformBaselineFlatness > max_waveform_baseline_fraction)
    )

    bombcell_class[ix] = 'noise'

    # assign well isolated units 
    ix = (
            (bombcell_class != 'noise') &
            (clusdat.nSpikes >= min_spike_num) &
            (clusdat.fractionRPVs_estimatedTauR <= max_refractory_period_violations) &
            (clusdat.rawAmplitude >= min_amp) &
            (clusdat.percentageSpikesMissing_gaussian <= max_percentage_spikes_missing) &
            (clusdat.signalToNoiseRatio >= minSNR) &
            (clusdat.presenceRatio >= min_presence_ratio)
    )

    bombcell_class[ix] = 'good'

    return bombcell_class


def get_subregions(regionNames, mode='Beryl'):
    def classify_SC_acronym(allen_acronym):
        if ('SCs' in allen_acronym) or ('SCo' in allen_acronym) or ('SCzo' in allen_acronym):
            my_acronym = 'SCs'
        elif ('SCi' in allen_acronym):
            my_acronym = 'SCi'
        elif ('SCd' in allen_acronym):
            my_acronym = 'SCd'
        else:
            my_acronym = 'nontarget'
        return my_acronym

    if mode == "Beryl":
        from Processing.pyhist.helpers.regions import BrainRegions
        reg = BrainRegions()
        regionNames[regionNames == 'unregistered'] = 'void'
        parentregions = reg.acronym2acronym(regionNames, mapping='Beryl')

    elif mode == 'Cosmos':
        from Processing.pyhist.helpers.regions import BrainRegions
        reg = BrainRegions()
        regionNames[regionNames == 'unregistered'] = 'void'
        parentregions = reg.acronym2acronym(regionNames, mapping='Cosmos')

    elif '3SC' == mode:
        parentregions = np.array([classify_SC_acronym(n) for n in regionNames])

    return parentregions


def is_rec_in_region(rec, region_name='SC', framework='ccf', min_fraction=.1, goodOnly=False, **bombcell_kwargs):
    """
    utility function to assess whether a recording contains neurons in a target region
    
    Parameters:
    ----------
    rec: pd.Series
        typically the output of queryExp.load_data. Must contain probe.clusters

    region_name: str
        name of the region in AllenAcronym to match. Does not need to be at the exact level in the hierarchy as the Allen name is written out 
        (e.g. SC will count SCs & SCm)
    min_fraction: float
        min fraction of neurons that must be in the region so that the recording passes

    Returns:
    -------
        :bool
        whether rec is in region with region_name or not 
    """

    clusters = rec.probe.clusters
    if goodOnly:
        bc_class = bombcell_sort_units(clusters, **bombcell_kwargs)  # %%
        is_good = bc_class == 'good'
        clusters = Bunch({k: clusters[k][is_good] for k in clusters.keys()})

    if min_fraction > 1:
        mode = 'number'
    else:
        mode = 'fraction'

    # check whether anatomy exists at all
    if 'mlapdv' not in list(clusters.keys()):
        is_region = False
    else:
        if framework == 'ccf':
            is_in_region = [region_name in x for x in clusters.brainLocationAcronyms_ccf_2017]
        else:
            region_names_in_framework = get_subregions(clusters.brainLocationAcronyms_ccf_2017, mode=framework)
            is_in_region = [x == region_name for x in region_names_in_framework]

        if (mode == 'fraction') & (np.mean(is_in_region) > min_fraction):
            is_region = True
        elif (mode == 'number') & (np.sum(is_in_region) > min_fraction):
            is_region = True

        else:
            is_region = False

    return is_region


def select_best_camera(rec, cam_hierarchy=['sideCam', 'frontCam', 'eyeCam']):
    """
    helper function that select a camera data based on the hierarchy and based on whether the camera data is avlaible at all 

    Parameters: 
    -----------
    rec: pd.Series 
        loaded ONE object 
    cam_hierarchy: list
        determines which camera view are prioritised

    """
    cam_checks = np.array([(hasattr(rec[cam].camera, 'ROIMotionEnergy') &
                            hasattr(rec[cam].camera, 'times')) if hasattr(rec, cam) else False for cam in
                           cam_hierarchy])
    if cam_checks.any():
        cam_idx = np.where(cam_checks)[0][0]
        used_camname = cam_hierarchy[cam_idx]
        cam = rec[used_camname].camera
    else:
        cam = None

    if hasattr(cam, 'ROIMotionEnergy'):
        if cam.ROIMotionEnergy.ndim == 2:
            cam.ROIMotionEnergy = (cam.ROIMotionEnergy[:, 0])

    return cam


def queryCSV(subject='all', expDate='all', expDef='all', expNum=None, checkIsSortedPyKS=None, checkEvents=None,
             checkSpikes=None, checkFrontCam=None, checkSideCam=None, checkEyeCam=None):
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
    checkIsSortedPyKS: None/str    
        if '2' only outputs

    checkEvents: None\str
        returns match to string if not none ('1', or '2')  
    checkSpikes: None/str
        returns match to string if not none ('1', or '2')  

    check_curation: bool
        only applies if unwrap_independent_probes is True
        whether the data has been curated in phy or not.

    Returns: 
    ----
    exp2checkList : pandas DataFrame 
        concatenated csv of requested experiments and its params 
    """

    mainCSVLoc = get_csv_location('main')
    mouseList = pd.read_csv(mainCSVLoc)
    # look for selected mice
    if 'allActive' in subject:
        mouse2checkList = mouseList[mouseList.IsActive == 1]['Subject']
    elif 'all' in subject:
        mouse2checkList = mouseList.Subject
    else:
        if not isinstance(subject, list):
            subject = [subject]
        mouse2checkList = mouseList[mouseList.Subject.isin(subject)]['Subject']

    exp2checkList = []
    for mm in mouse2checkList:
        mouse_csv = get_csv_location(mm)
        if mouse_csv.is_file():
            expList = pd.read_csv(mouse_csv, dtype='str')
            expList.expDate = [date.replace('_', '-').lower() for date in expList.expDate.values]
            if 'all' not in expDef:
                if not isinstance(expDef, list):
                    expDef = [expDef]

                is_selected_defs = np.concatenate(
                    [expList.expDef.str.contains(curr_expDef).values[np.newaxis] for curr_expDef in expDef]).sum(
                    axis=0).astype('bool')
                expList = expList[is_selected_defs]

            if 'all' not in expDate:
                # dealing with the call of posImplant based on the main csv. Otherwise one is able to use any date they would like 
                if ('postImplant' in expDate):
                    implant_date = mouseList[mouseList.Subject == mm].P0_implantDate
                    # check whether mouse was implanted at all or not.
                    if ~implant_date.isnull().values[0] & (expList.size > 0):
                        implant_date = implant_date.values[0]
                        implant_date = implant_date.replace('_', '-').lower()
                        implant_date_range = implant_date + ':' + expList.expDate.iloc[-1]
                        selected_dates = check_date_selection(implant_date_range, expList.expDate)
                    else:
                        print('%s was not implanted or did not have the requested type of exps.' % mm)
                        selected_dates = np.zeros(expList.expDate.size).astype('bool')

                    expList = expList[selected_dates]

                elif ('last' in expDate):
                    # this only selects the last experiment done on the given animal
                    how_many_days = int(expDate.split('last')[1])
                    expList = expList.iloc[-how_many_days:]

                else:
                    selected_dates = check_date_selection(expDate, expList.expDate)
                    expList = expList[selected_dates]

            if expNum:
                expNum = (np.array(expNum)).astype('str')
                _, idx, _ = np.intersect1d(expList.expNum.to_numpy(), expNum, return_indices=True)
                expList = expList.iloc[idx]

                # add mouse name to list
            expList['subject'] = mm

            exp2checkList.append(expList)

    if len(exp2checkList) > 0:
        exp2checkList = pd.concat(exp2checkList)
        # re-index
        exp2checkList = exp2checkList.reset_index(drop=True)

    else:
        print('you did not call any experiments.')
        exp2checkList = None

    if len(exp2checkList) > 0:
        if checkIsSortedPyKS is not None:
            # nan means we should not have ephys. So we drop nan columns
            exp2checkList = exp2checkList[exp2checkList['issortedPyKS'].notna()]
            to_keep_column = np.array([checkIsSortedPyKS in rec.issortedPyKS for _, rec in exp2checkList.iterrows()])
            exp2checkList = exp2checkList[to_keep_column]

        if checkEvents is not None:
            exp2checkList = exp2checkList[exp2checkList['extractEvents'].notna()]
            to_keep_column = np.array([checkEvents in rec.extractEvents for _, rec in exp2checkList.iterrows()])
            exp2checkList = exp2checkList[to_keep_column]

        if checkSpikes is not None:
            exp2checkList = exp2checkList[exp2checkList['extractSpikes'].notna()]
            to_keep_column = np.array([checkSpikes in rec.extractSpikes for _, rec in exp2checkList.iterrows()])
            exp2checkList = exp2checkList[to_keep_column]

        if checkFrontCam is not None:
            exp2checkList = exp2checkList[
                exp2checkList['alignFrontCam'].notna() & exp2checkList['fMapFrontCam'].notna()]
            to_keep_column = np.array(
                [(checkFrontCam in rec.alignFrontCam) & (checkFrontCam in rec.fMapFrontCam) for _, rec in
                 exp2checkList.iterrows()])
            exp2checkList = exp2checkList[to_keep_column]

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


def load_ONE_object(collection_folder, object, attributes='all'):
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
    object_names = [re.split(r"\.", file.name)[0] for file in file_names]
    attribute_names = [re.split(r"\.", file.name)[1] for file in file_names]
    extensions = [re.split(r"\.", file.name)[-1] for file in file_names]

    if 'all' in attributes:
        attributes = attribute_names

    output = {}

    for f, o, a, e in zip(file_names, object_names, attribute_names, extensions):
        # normally don't load any large data with this loader 
        if 'npy' in e:
            if a in attributes:
                tempload = np.load(f)
                if (tempload.ndim == 2):
                    if (tempload.shape[1] == 1):  # if its the stupid matlab format, ravel
                        output[a] = tempload[:, 0]
                    else:
                        output[a] = tempload
                else:
                    output[a] = tempload

        elif 'pqt' in e:
            if a in attributes:  # now I just not load the largeData
                tempload = pd.read_parquet(f)
                tempload = tempload.to_dict('list')

                for k in tempload.keys():
                    output[k] = np.array(tempload[k])

        elif 'json' in e:
            if a in attributes:
                tempload = open(f, )
                tempload = json.load(tempload)
                tempload = Path(tempload)
                output[a] = tempload

    output = Bunch(output)

    return output


def load_data(recordings=None,
              data_name_dict=None,
              unwrap_probes=False,
              merge_probes=False,
              region_selection=None,
              filter_unique_shank_positions=False,
              cam_hierarchy=None, **kwargs):
    """
    Paramters: 
    -------------
    recrordings: pd.df
        csv that contains the list of recordings to load.
        In particular the csv ought to contain the column "expFolder" such that points to the parent folder of the ONE_preproc
        if None, then function uses the queryCSVs
         
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
    
    unwrap_probes: bool
        returns exp2checkList with a probe tag where so each row is separate probe as opposed to a session 
    merge_probes: bool
        returns a exp2checkList with a single column of ehys data, where data from both probes are merged into a single Bunch of clusters and spikes
        where clusterIDs from the 2nd probe get 10k added to them

    Returns: 
    -------------
    pd.DataFrame 
        collections  requested by data_name_dict are added as columns  to the original csvs.   

    Todo: implement cond loading,default params
        
    """
    if recordings is None:
        recordings = queryCSV(**kwargs)
    else:
        recordings = recordings[['subject', 'expDate', 'expDef', 'expFolder']]

    if data_name_dict:

        collections = list(data_name_dict.keys())
        for collection in collections:
            recordings[collection] = None
            for idx, rec in recordings.iterrows():
                # to do -- make it dependent on whether the extraction was done correctly 

                exp_folder = Path(rec.expFolder)
                if 'raw' in collection and 'probe' in collection:
                    probe_name = re.findall('probe\d', collection)[0]
                    probe_collection_folder = exp_folder / 'ONE_preproc' / probe_name
                    raw_path = list((probe_collection_folder).glob('_av_rawephys.path.*.json'))
                    if len(raw_path) == 1:
                        # then a matching ephys file is found
                        ev_collection_folder = open(raw_path[0], )
                        ev_collection_folder = json.load(ev_collection_folder)
                        ev_collection_folder = Path(ev_collection_folder)
                    else:
                        ev_collection_folder = probe_collection_folder
                else:
                    ev_collection_folder = exp_folder / 'ONE_preproc' / collection

                objects = {}
                for object in data_name_dict[collection]:
                    objects[object] = load_ONE_object(ev_collection_folder, object,
                                                      attributes=data_name_dict[collection][object])
                objects = Bunch(objects)
                recordings.loc[idx][collection] = objects

        ### dealing with camera data selection ####
        if cam_hierarchy:
            camdat = [select_best_camera(rec, cam_hierarchy=cam_hierarchy) for _, rec in recordings.iterrows()]
            recordings['camera'] = camdat
            recordings = recordings[~recordings.camera.isna()]

        ### ####### deling with extra arguments that further format the data ######
        # merge probes
        # an optional argument for when there are numerous datasets available for probes, we just merge the data

        if unwrap_probes or merge_probes:
            expected_probe_no = ((recordings.extractSpikes.str.len() / 2) + 0.5)
            expected_probe_no[np.isnan(expected_probe_no)] = 0
            expected_probe_no = expected_probe_no.astype(int)

            recordings['expected_probe_no'] = expected_probe_no

            old_columns = recordings.columns.values
            keep_columns = np.setdiff1d(old_columns, ['ephysPathProbe0', 'ephysPathProbe1', 'probe0', 'probe1'])

            if unwrap_probes & (~merge_probes):
                # this is the mode when we crate a different row for each probe
                rec_list = []
                for _, rec in recordings.iterrows():
                    for p_no in range(rec.expected_probe_no):
                        string_idx = (p_no) * 2
                        if (rec.extractSpikes[int(string_idx)] == '1'):
                            myrec = rec[keep_columns]
                            myrec['probeID'] = 'probe%s' % p_no
                            myrec['probe'] = rec['probe%s' % p_no]
                            ephysPath = rec['ephysPathProbe%s' % p_no]
                            myrec['ephysPath'] = ephysPath

                            curated_fileMark = Path(ephysPath)
                            curated_fileMark = curated_fileMark / 'pyKS\output\cluster_info.tsv'
                            myrec['is_curated'] = curated_fileMark.is_file()

                            rec_list.append(myrec)

                    recordings = pd.DataFrame(rec_list, columns=np.concatenate(
                        (keep_columns, ['probeID', 'probe', 'ephysPath', 'is_curated'])))

                if filter_unique_shank_positions:
                    # only do this with chronic insertions
                    botrow_positions = np.arange(8) * 720
                    botrow_targets = [
                        botrow_positions[np.argmin(np.abs(botrow_positions - min(rec.probe.clusters.depths)))] for
                        _, rec in recordings.iterrows()]
                    recordings['botrow'] = botrow_targets

                    acute_recs = recordings[recordings.rigName == 'lilrig-stim']
                    chronic_recs = recordings[recordings.rigName != 'lilrig-stim']

                    chronic_recs = chronic_recs.drop_duplicates(subset=['subject', 'probeID', 'botrow'])
                    recordings = pd.concat((chronic_recs, acute_recs))



            elif (~unwrap_probes) & merge_probes:
                rec_list = []

                for _, rec in recordings.iterrows():
                    myrec = rec[keep_columns]

                    is_probe0 = hasattr(rec.probe0.spikes, 'amps')
                    is_probe1 = hasattr(rec.probe1.spikes, 'amps')

                    # add new data about probe ID to both spikes and clusters 
                    if is_probe0:
                        rec.probe0.spikes['probeID'] = np.zeros(rec.probe0.spikes.amps.size)
                        rec.probe0.clusters['probeID'] = np.zeros(rec.probe0.clusters.amps.size)

                    if is_probe1:
                        rec.probe1.spikes['probeID'] = np.ones(rec.probe1.spikes.amps.size)
                        rec.probe1.clusters['probeID'] = np.ones(rec.probe1.clusters.amps.size)

                    # redo the _av_IDs of probe 1 if needed
                    if is_probe0 & is_probe1:
                        rec.probe1.clusters._av_IDs = rec.probe1.clusters._av_IDs + 1000
                        rec.probe1.spikes.clusters = rec.probe1.spikes.clusters + 1000

                        sp, cl = rec.probe0.spikes.keys(), rec.probe0.clusters.keys()

                        new_spikes = Bunch(
                            {k: np.concatenate((rec.probe0.spikes[k], rec.probe1.spikes[k])) for k in sp})
                        new_clusters = Bunch(
                            {k: np.concatenate((rec.probe0.clusters[k], rec.probe1.clusters[k])) for k in cl})

                        myrec['probe'] = Bunch({'spikes': new_spikes, 'clusters': new_clusters})
                    elif is_probe0 & ~is_probe1:
                        myrec['probe'] = rec.probe0
                    elif ~is_probe0 & is_probe1:
                        myrec['probe'] = rec.probe1

                    rec_list.append(myrec)

                recordings = pd.DataFrame(rec_list, columns=np.concatenate((keep_columns, ['probe'])))

                # first we identify whether there are two probes
                # then we process the 2nd probes data such that it can be disambiguated from the first probe
                # process the 1st probe
                # merge

            if region_selection is not None:
                keep_rec_region = [is_rec_in_region(rec, **region_selection) for _, rec in recordings.iterrows()]
                recordings = recordings[keep_rec_region]

        # give each recording a unique ID

    return recordings


def load_ephys_independent_probes(probe='probe0', ephys_dict={'spikes': ['times', 'clusters']}, add_dict=None,
                                  raw_ephys_dict=None, **kwargs):
    """
    This is a helper function to single probe data. 
    Parameters:
    -----------
    probe: str
        probe ID: either probe0 or probe1
    ephys_dict: dict -- a must
        what you want to call from probe from the ONE folder
    raw_ephys_dict: None/dict
        what you want to call from the raw IBL folder 
    add_dict: None/dict
        anything else (ONE) you want to call -- events/camera etc.
    additional kwargs: usual input of load data/queryCSV
        such as subject,expDate,expNum
    """
    d = {probe: ephys_dict}  # requirement to call some ephys
    # add any additional dicts
    if raw_ephys_dict:
        d.update({('%s_raw' % probe): raw_ephys_dict})
    if add_dict:
        d.update(add_dict)

    r = load_data(data_name_dict=d, **kwargs)

    r = r.rename(columns={probe: 'probe'})
    r = r.rename(columns={('%s_raw' % probe): 'ibl'})

    return r


def format_events(ev, reverse_opto=False):
    if hasattr(ev, 'stim_visContrast'):
        ev.stim_visContrast = np.round(ev.stim_visContrast, 2)
    if hasattr(ev, 'stim_audAmplitude'):
        ev.stim_audAmplitude = np.round(ev.stim_audAmplitude, 2)

        amps = np.unique(ev.stim_audAmplitude)

        if (
        amps[amps > 0]).size == 1:  # if there is only one amp then the usual way of calculating audDiff etc is valid
            ev.visDiff = ev.stim_visContrast * np.sign(ev.stim_visAzimuth)
            ev.visDiff[np.isnan(ev.visDiff)] = 0
            ev.audDiff = np.sign(ev.stim_audAzimuth)

    if hasattr(ev, 'timeline_choiceMoveOn'):
        ev.rt = ev.timeline_choiceMoveOn - np.nanmin(
            np.concatenate([ev.timeline_audPeriodOn[:, np.newaxis], ev.timeline_visPeriodOn[:, np.newaxis]], axis=1),
            axis=1)
        ev.rt_aud = ev.timeline_choiceMoveOn - ev.timeline_audPeriodOn
        ev.first_move_time = ev.timeline_firstMoveOn - np.nanmin(
            np.concatenate([ev.timeline_audPeriodOn[:, np.newaxis], ev.timeline_visPeriodOn[:, np.newaxis]], axis=1),
            axis=1)
    if hasattr(ev, 'is_laserTrial') & hasattr(ev, 'stim_laser1_power') & hasattr(ev, 'stim_laser2_power'):
        ev.laser_power = (ev.stim_laser1_power + ev.stim_laser2_power).astype('int')
        ev.laser_power_signed = (ev.laser_power * ev.stim_laserPosition)
        if reverse_opto & ~(np.unique(ev.laser_power_signed > 0).any()):
            # if we call this than if within the session the opto is only on the left then we reverse the azimuth and choices on that session
            ev.stim_audAzimuth = ev.stim_audAzimuth * -1
            ev.stim_visAzimuth = ev.stim_visAzimuth * -1
            ev.timeline_choiceMoveDir = ((ev.timeline_choiceMoveDir - 1.5) * -1) + 1.5

    return ev


def format_cluster_data(clusters):
    """
    this is a helper that further formats the cluster data, mostly with adding information about the anatomy and quality metrics
    
    """
    clusInfo = {k: clusters[k] for k in clusters.keys() if clusters[k].ndim == 1}
    clusInfo = pd.DataFrame.from_dict(clusInfo)
    # clusInfo = clusInfo.set_index('_av_IDs',drop=False)

    colnames = list(clusters.keys())
    if 'mlapdv' in colnames:
        # we could add the raw, but for now, I won't actually
        clusInfo['ml'] = clusters.mlapdv[:, 0]
        clusInfo['ap'] = clusters.mlapdv[:, 1]
        clusInfo['dv'] = clusters.mlapdv[:, 2]
        clusInfo['hemi'] = np.sign(clusInfo.ml - 5600)

    else:
        clusInfo['ml'] = np.nan
        clusInfo['ap'] = np.nan
        clusInfo['dv'] = np.nan
        clusInfo['hemi'] = np.nan
        clusInfo['brainLocationAcronyms_ccf_2017'] = 'unregistered'
        clusInfo['brainLocationIds_ccf_2017'] = np.nan

    if 'phy_clusterID' not in colnames:
        clusInfo['phy_clusterID'] = clusInfo.cluster_id

    from Processing.pyhist.helpers.regions import BrainRegions
    br = BrainRegions()
    bc_class = bombcell_sort_units(clusInfo)
    clusInfo['bombcell_class'] = bc_class
    clusInfo['is_good'] = bc_class == 'good'
    clusInfo.brainLocationAcronyms_ccf_2017[
        clusInfo.brainLocationAcronyms_ccf_2017 == 'unregistered'] = 'void'  # this is just so that the berylacronymconversion does something good
    clusInfo['BerylAcronym'] = br.acronym2acronym(clusInfo.brainLocationAcronyms_ccf_2017, mapping='Beryl')

    return clusInfo


def simplify_recdat(recording, probe='probe', reverse_opto=False, cam_hierarchy=['sideCam', 'frontCam', 'eyeCam']):
    """
    this is the most standarising loader. Allows standardisation of numerous sessions etc. 
    spits out the event,spike etc bunches with one line
    allows for quicker calling of data in a single experiment
    also allows calculations of extra parameters that vary session to session, utilised for mutliSpaceWorld 
    such that 
    A) we calculate reaction times 
    B) we reverse opto to do ipsi.contra calculations. In this case, 
        stim_audAzimuth/stim_visAzimuth: +ve: ipsi, -ve contra
        timeline_choiceMoveDir 1 contra, 2 ipsi

    Parameters: 
    -----------
    recording: pd.Series
        details of recording as outputted by load date
    probe: str
        name of the probe in recording pd.Series that we want to spit out. 
        This works, of probe = 'probe', i.e. when load_data was done by aleady splitting the probes
        Or if the probes were loaded with their ID 'probe0' or 'probe1'
    reverse_opto:bool
    cam_hierarchy: list
        we load a single camera's data. Cam_hierarchy will determine which one exactly.I.e. if 1st exist, load 1st, if not load2nd etc.
        so cam_hierarchy is a list of names. 

    Retruns:
    --------
        Bunch,Bunch,pd.dataFrame,Bunch
        for ev,spikes,clusters & channels
        if it does not exist, we will out None
    """
    ev, spikes, clusters, channels, cam = None, None, None, None, None

    if hasattr(recording, 'events'):
        ev = recording.events._av_trials
        ev = format_events(ev, reverse_opto=reverse_opto)

    if hasattr(recording, probe):
        p_dat = recording[probe]
        if hasattr(p_dat, 'spikes'):
            spikes = p_dat.spikes

        if hasattr(p_dat, 'clusters'):
            clusters = format_cluster_data(p_dat.clusters)

        if hasattr(p_dat, 'channels'):
            channels = p_dat.channels

    cam = select_best_camera(recording, cam_hierarchy=cam_hierarchy)

    return (ev, spikes, clusters, channels, cam)


def get_recorded_channel_position(channels):
    """
    todo: get IBL channels parameter. I think this needs to be implemented on the PinkRig level.
    """
    if not channels:
        xrange, yrange = None, None
    else:
        xcoords = channels.localCoordinates[:, 0]
        ycoords = channels.localCoordinates[:, 1]

        # if the probe is 3B, pyKS for some reason starts indexing from 1 depth higher (not 0)
        # to be fair that might be more fair, because the tip needs to be calculated to the anatomy 
        # alas who cares.
        if np.max(np.diff(ycoords)) == 20:
            # 3B probe
            ycoords = ycoords - ycoords[0]

        xrange = (np.min(xcoords), np.max(xcoords))
        yrange = (np.min(ycoords), np.max(ycoords))

    return (xrange, yrange)


def load_active_and_passive(rec_info):
    """
    function to load active and passive data together using load_ephys_independent probes (i.e. can call one probe at a time) 

    Parameters: 
    -----------
    rec_info (dict)
        will be passed as kwarg to load_ephys_independent probes
    
    Return: Bunch

    """
    ephys_dict = {'spikes': 'all', 'clusters': 'all'}
    other_ = {'events': {'_av_trials': 'table'}}

    session_types = ['postactive', 'multiSpaceWorld']

    dat = {}
    for sess in session_types:
        recordings = load_ephys_independent_probes(ephys_dict=ephys_dict, add_dict=other_, expDef=sess, **rec_info)

        # select the longest etc. recording (there might be several)
        recordings = recordings.iloc[np.argmax(recordings.expDuration)]

        events, spikes, clusters, _, cam = simplify_recdat(recordings, probe='probe')
        d = Bunch({'events': events, 'spikes': spikes, 'clusters': clusters})
        dat[sess] = d
    dat = Bunch(dat)

    return dat


def concatenate_events(recordings, filter_type=None):
    """
    function to concatenate events from pd. df recordings output (i.e. output of load data)

    Parameters:
    -----------
    recordings: pd.df 
        output of csv_queryExp.load_data
    filter_type: str
        codenames for criteria to throw sessions away. Options implemented: 
        'finalStage'
        'opto'


    """

    should_reverse_opto = False

    if 'optoUniBoth':
        should_reverse_opto = True

        # maybe I could write some function titled concatenate events
    ev, _, _, _, _ = zip(*[simplify_recdat(rec, reverse_opto=should_reverse_opto) for _, rec in recordings.iterrows()])

    # write in subject ID and sessionID into the ev in long fofor e in ev]
    for (i, e), s in zip(enumerate(ev), recordings.subject):
        ev[i].sessionID = np.ones(e.is_blankTrial.size) * i
        ev[i].subject = np.array([s for x in range(e.is_blankTrial.size)])
    # filter
    if filter_type:
        if 'final_stage' in filter_type:
            is_kept_session = [e.is_conflictTrial.sum() > 5 for e in ev]
        elif 'optoUniBoth' in filter_type:
            is_kept_session = [
                (np.sum(e.is_laserTrial) > 0) & (np.abs(e.stim_laserPosition) == 1).any() & (e.laser_power == 17).any()
                for e in ev]
        elif 'optoBi' in filter_type:
            is_kept_session = [(np.sum(e.is_laserTrial) > 0) & (np.abs(e.stim_laserPosition) == 0).any() for e in ev]
    else:
        is_kept_session = np.ones(len(ev)).astype('bool')

    ev = list(compress(ev, is_kept_session))
    # concatenate
    ev_keys = list(ev[0].keys())
    ev = Bunch({k: np.concatenate([e[k] for e in ev]) for k in ev_keys})

    return ev


def queryGood():
    pass
