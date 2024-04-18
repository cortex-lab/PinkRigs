from pathlib import Path
import json 

def save_dict_to_json(dict,path):
    """
    util function to save dictionaries to json
    """
    message = json.dumps(dict)
    errfile = open(path,"w")
    errfile.write(message)
    errfile.close()



def get_subfolders(folder_path):
    folder = Path(folder_path)
    subfolders = [subfolder for subfolder in folder.iterdir() if subfolder.is_dir()]
    return subfolders

