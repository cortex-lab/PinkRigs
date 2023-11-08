import numpy as np

def add_gauss_to_mlapdv(allen_pos_mlapdv,ml=50,ap=30,dv=0):
    """
    function to add gaussian noise to allen atlas locations, particularly along the inserion axis 
    sd of each axes is in arg. 
    """
    
    noise = np.random.normal(loc=0,scale=ml,size=(allen_pos_mlapdv.shape[0],))
    allen_pos_mlapdv[:,0] = allen_pos_mlapdv[:,0] + noise
    noise = np.random.normal(loc=0,scale=ap,size=(allen_pos_mlapdv.shape[0],))
    allen_pos_mlapdv[:,1] = allen_pos_mlapdv[:,1] + noise
    noise = np.random.normal(loc=0,scale=dv,size=(allen_pos_mlapdv.shape[0],))
    allen_pos_mlapdv[:,2] = allen_pos_mlapdv[:,2] + noise
    return allen_pos_mlapdv

def add_gauss_to_apdvml(allen_pos_apdvml,ml=50,ap=30,dv=0):
    """
    function to add gaussian noise to allen atlas locations, particularly along the inserion axis 
    sd of each axes is in arg. 
    """
    np.random.seed(0)
    noise = np.random.normal(loc=0,scale=ml,size=(allen_pos_apdvml.shape[0],))
    allen_pos_apdvml[:,2] = allen_pos_apdvml[:,2] + noise
    noise = np.random.normal(loc=0,scale=ap,size=(allen_pos_apdvml.shape[0],))
    allen_pos_apdvml[:,0] = allen_pos_apdvml[:,0] + noise
    noise = np.random.normal(loc=0,scale=dv,size=(allen_pos_apdvml.shape[0],))
    allen_pos_apdvml[:,1] = allen_pos_apdvml[:,1] + noise
    return allen_pos_apdvml

def add_gauss_to_dvmlap(allen_pos_dvmlap,ml=50,ap=30,dv=0):
    """
    function to add gaussian noise to allen atlas locations, particularly along the inserion axis 
    sd of each axes is in arg. 
    """
    np.random.seed(0)
    noise = np.random.normal(loc=0,scale=ml,size=(allen_pos_dvmlap.shape[0],))
    allen_pos_dvmlap[:,1] = allen_pos_dvmlap[:,1] + noise
    noise = np.random.normal(loc=0,scale=ap,size=(allen_pos_dvmlap.shape[0],))
    allen_pos_dvmlap[:,2] = allen_pos_dvmlap[:,2] + noise
    noise = np.random.normal(loc=0,scale=dv,size=(allen_pos_dvmlap.shape[0],))
    allen_pos_dvmlap[:,0] = allen_pos_dvmlap[:,0] + noise
    return allen_pos_dvmlap