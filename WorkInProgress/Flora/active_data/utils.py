import pickle

def save_pickle(mydict,path):
    with open(path.__str__(), 'wb') as f:
        pickle.dump(mydict,f,pickle.HIGHEST_PROTOCOL)

def read_pickle(path):
    with open(path.__str__(),'rb') as f:
        obj = pickle.load(f)
    return obj
