import glob,sys
from pathlib import Path 

# pinkRigs specific functions
pinkRig_path= glob.glob(r'C:\Users\*\Documents\Github\PinkRigs')
pinkRig_path = Path(pinkRig_path[0])
sys.path.insert(0, (pinkRig_path.__str__()))

from Admin.csv_queryExp import queryCSV

def preproc_ephys(func=None,funckwargs=None,**kwargs): 
    """
    this function allows the processing of ephys using queryExp outputs, i.e. for given expNums
    
    Parameters: 
    -----------
    func: str
        corresponds to the type of function you wish to execute.
        current options are: 
        compress
        decompress
        ibl_anat -- ibl extraction with anatomy -- have to be done in iblenv
        ibl -- ibl extraction without anatomy -- have to be done in iblenv
        kilo -- pyKS, have to be done in pyKS2 env
        clus_anat -- add cluster anatomy location after alignment has been performed
    fuckwargs(optional): dict
        if you want to pass down arguments for the functions called by func
    kwargs
        for queryCSV. standard kwargs are subject,expDate,expNum,expDef
    example uses in manual.ipynb
    
    """
    # here imports need to be func specific as they are in different environments
    recordings = queryCSV(**kwargs)
    ephys_paths = recordings.ephysPathProbe0.dropna().to_list() + recordings.ephysPathProbe1.dropna().to_list()
    ephys_paths = [Path(p) for p in ephys_paths]


    for rec in ephys_paths:
        # read the corresponding ephys files
        uncompressed_path = list(rec.glob('**\*.bin'))
        compressed_path = list(rec.glob('**\*.cbin'))
        ch_path = list(rec.glob('**\*.ch'))

        if func == 'compress':
            from Admin.helpers.compress_data import mainCompress
            # find whether there is a uncompressed filed
            if len(uncompressed_path)!=1:
                print('%s is already compressed.' % rec.__str__())
            else:
                print('compressing ...')
                mainCompress(binFile=(uncompressed_path[0]).__str__())

        if 'decompress' in func:
            from Admin.helpers.decompress_data import mainDecompress
            # find whether there is a uncompressed filed
            if len(compressed_path)!=1:
                print('%s is not compressed.' % rec.__str__())
                if len(ch_path)!=1:
                    print('%s - we cannot find the .ch file.' % rec.__str__())
            else:
                print('compressing ...')
                mainDecompress(
                    cbinFile=(compressed_path[0]).__str__(),
                    chFile=(ch_path[0]).__str__()
                    )
                                    
        if 'ibl_anat' in func:
            from Processing.pykilo.convert_to_ibl_format import add_anat_to_ibl_format
            if not funckwargs:
                add_anat_to_ibl_format(rec,recompute=False)
            else:
                add_anat_to_ibl_format(rec,**funckwargs)

        elif 'ibl' in func:
            from Processing.pykilo.convert_to_ibl_format import ks_to_ibl_format
            ks_to_ibl_format(rec,ks_folder='pyKS',recompute=True)

        elif 'kilo' in func:
            from Processing.pykilo.run_pyKS import run_pyKS_single_file
            if not funckwargs:
                run_pyKS_single_file(compressed_path[0],recompute_errored_sorting = True,resort = False,bin_file_extension ='cbin')
            else: 
                run_pyKS_single_file(compressed_path[0],**funckwargs)

        elif 'clus_anat' in func:
            from Processing.pyhist.assign_clusters_to_atlas import save_out_cluster_location
            save_out_cluster_location(rec)