# generic packages 
import glob,re,pickle
from pathlib import Path
import numpy as np
#ephys data specific packages 
from ReadSGLXData.readSGLX import readMeta

def match_recordings(Subject='AV005'):
    """
    gets all .cbin files within the folder 
    and matches them based on probe serial number and imro table
    read in from the meta file 

    inputs:
        Subject: str, subject name
    outputs: 
        no returns 
        save dictionary of matched .cbin lists in a pcl put on server

    todo: 
        - deal with data coming from different servers 

    """
    serverroot = r'\\zinu.cortexlab.net\Subjects'
    ephyspath = r'%s\%s\**\ephys\**\*.cbin' % (serverroot,Subject)

    # main stitchedPyKS dir 

    saveroot = Path(r'%s\%s\stitchedPyKS' % (serverroot,Subject))
    saveroot.mkdir(parents=False, exist_ok=True)  

    sn_list = []
    shank_map_list = []
    ephysdat_list = np.array(glob.glob(ephyspath,recursive=True))

    for binfile in ephysdat_list: 
        meta = readMeta(Path(binfile))
        probe_sn = meta['imDatPrb_sn']
        shank_map = meta['snsShankMap']
        sn_list.append(probe_sn)
        shank_map_list.append(shank_map)

    sn_list = np.array(sn_list)
    shank_map_list = np.array(shank_map_list)

    # get unique probe serial numbers form each 

    stitched_sorting_data = {}

    probes = np.unique(sn_list)
    for myprobe in probes:
        is_myprobe= [sn==myprobe for sn in sn_list] 
        myprobe_shankmap_list = shank_map_list[is_myprobe]

        unique_shankmaps = np.unique(myprobe_shankmap_list)
        for myimro in unique_shankmaps:
            is_myimro = [imro==myimro for imro in myprobe_shankmap_list]
            matching_ephys_list = ephysdat_list[is_myprobe][is_myimro]

            #get the namestring for this imro
            # to give the unique shankmaps some more readable name 
            channel_data = re.findall(r"\d+:\d+:\d+:\d+",myimro)
            shank_idx = np.array([np.array(re.split(':',chandat)).astype('int')[0] for chandat in channel_data])
            depth_idx = np.array([np.array(re.split(':',chandat)).astype('int')[2] for chandat in channel_data])

            shankstring = "".join(list((np.unique(shank_idx).astype('str'))))
            botrow = np.min(depth_idx)
            sn_imro_string = 'serial%s_shank%s_botrow%.0f' % (myprobe,shankstring,botrow)

            # create main directory      
            stitched_sorting_data[sn_imro_string] = matching_ephys_list


    f=open(saveroot / 'matched_imros.pcl','wb')
    pickle.dump(stitched_sorting_data,f)
    f.close()