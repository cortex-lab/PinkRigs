from pathlib import Path

def get_paths(set_name):
    """
    Creates standard path structure for data_management and can recall Paths
    Parameters: 
        set_name: str
            identifier of the dataset that points to the raw data 

    Returns:
        basepath,formatted_path,savepath 
        pathlib.Paths 
        raw data, processed data for fitting, results of fitting
    """
    basepath = Path(r'D:\LogRegression\%s' % set_name)
    formatted_path = basepath / 'formatted'
    formatted_path.mkdir(parents=False,exist_ok=True)
    savepath = formatted_path / 'fit_results'
    savepath.mkdir(parents=False,exist_ok=True)

    return basepath,formatted_path,savepath