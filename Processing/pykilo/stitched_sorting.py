# generic packages 
import sys,glob,re,pickle,json
from pathlib import Path
import numpy as np
#ephys data specific packages 
from Processing.pykilo.ReadSGLXData.readSGLX import readMeta
from pykilosort import run, add_default_handler, neuropixel_probe_from_metafile

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
    server = r'\\zinu.cortexlab.net\Subjects'
    ephyspath = r'%s\%s\**\ephys\**\*.cbin' % (server,Subject)

    # main stitchedPyKS dir 

    saveroot = Path(r'%s\%s\stitchedPyKS' % (server,Subject))
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

def run_stitchPyKS(Subject='AV005'):
    """
    function to read in the pickle and sort using the stitched recordings
    """

    # if the recordings have not been aready matched, match it 
    server = r'\\zinu.cortexlab.net\Subjects'
    stitch_root= Path(r'%s\%s\stitchedPyKS' % (server,Subject))
    matched_data_path = list((stitch_root).glob('matched_imros.pcl'))
    if not matched_data_path: 
        match_recordings(Subject=Subject)
    # recheck
    matched_data_path = list((stitch_root).glob('matched_imros.pcl'))[0]

    # read in the pickle
    f=open(matched_data_path,'rb')
    matched_dat=pickle.load(f)
    f.close()

    for config in matched_dat.keys(): 
        if len(matched_dat[config])>1:
            # sort some stitched stuff 
            print('we have something to sort!!')
            try:  
                output_dir  = stitch_root / config
                output_dir.mkdir(parents=False, exist_ok=True)

                check_dir = output_dir / 'output'
                check_sorted = list(check_dir.glob('spike_times.npy'))
                # if it has not been sorted alr
                if not check_sorted:           

                    input_dirs = matched_dat[config]
                    channel_map = neuropixel_probe_from_metafile(input_dirs[0])
                    # get the channelmap for the example config data 

                    input_dirs = [Path(ephysfilepath) for ephysfilepath in input_dirs]

                    print('starting sorting ...')
                    add_default_handler(level='INFO') # print output as the algorithm runs
                    run(input_dirs, probe=channel_map, low_memory=True, dir_path=output_dir)

            except: 
                # save error message 
                exc_type, exc_obj, exc_tb = sys.exc_info()
        
                errdict = {
                    'err_type:': str(exc_type), 
                    'err_message': str(exc_obj),
                    'traceback': str(exc_tb)
                    }

                errmessage = json.dumps(errdict)
                
                errfile = open(output_dir / 'pyKS_error.json',"w")
                errfile.write(errmessage)
                errfile.close()

if __name__ == "__main__":
   #run_stitchPyKS(Subject=sys.argv[1])
    run_stitchPyKS(Subject='AV008')