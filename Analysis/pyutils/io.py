import json 

def save_dict_to_json(dict,path):
    """
    util function to save dictionaries to json
    """
    message = json.dumps(dict)
    errfile = open(path,"w")
    errfile.write(message)
    errfile.close()