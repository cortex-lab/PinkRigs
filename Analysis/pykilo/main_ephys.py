import json,glob,sys
from pathlib import Path 

# pinkRigs specific functions
pinkRig_path= glob.glob(r'C:\Users\*\Documents\Github\PinkRigs')
pinkRig_path = Path(pinkRig_path[0])
sys.path.insert(0, (pinkRig_path.__str__()))

from Admin.csv_queryExp import queryCSV

def query_and_process_ephys(func=None,funckwargs=None,**kwargs): 
    """
    this function allows the processing of ephys using queryExp outputs, i.e. for given expNums

    """
    # here imports need to be func specific as they are in different environments

    recordings = queryCSV(**kwargs)
    for _,r in recordings.iterrows():
    # read corresponding ephys files 
        corresponding_ephys_json= list((Path(r.expFolder)).glob('ONE_preproc\probe*\_av_rawephys.path*.json'))
        if len(corresponding_ephys_json)==0:
            print('there is probably no ONE file. This func only works if there is one.')
        for rec_path in corresponding_ephys_json: 
                # open json to get the actual path
                rec = open(rec_path,)
                rec = json.load(rec)
                rec = Path(rec)

                if 'ibl_anat' in func:
                    from Analysis.pykilo.convert_to_ibl_format import add_anat_to_ibl_format
                    if not funckwargs:
                        add_anat_to_ibl_format(rec.parents[2],recompute=False)
                    else:
                        add_anat_to_ibl_format(rec.parents[2],**funckwargs)
                elif 'ibl' in func:
                    from Analysis.pykilo.convert_to_ibl_format import ks_to_ibl_format
                    ks_to_ibl_format(rec,ks_folder='pyKS',recompute=True)
                elif 'kilo' in func:
                    from Analysis.pykilo.run_pyKS import run_pyKS_single_file
                    run_pyKS_single_file(rec,recompute_errored_sorting = True,resort = False,bin_file_extension ='cbin')
                elif 'clus_anat' in func:
                    from Analysis.pyhist.assign_clusters_to_atlas import save_out_cluster_location
                    save_out_cluster_location(rec)