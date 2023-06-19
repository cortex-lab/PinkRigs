import json

def save_error_message(output_filename,err_type='None',err_message='None',err_traceback='None'): 
    """
    function to save sys.exec_info output into a json 
    Parameters: 
    -------------
    output_filename: pathlib.path
    err_type: obj
        type of error 
    err_message: obj
        error message
    err_traceback: obj
        traceback message
    """
    errdict = {
    'err_type:': str(err_type), 
    'err_message': str(err_message),
    'traceback': str(err_traceback)
    }

    errmessage = json.dumps(errdict)

    errfile = open(output_filename,"w")
    errfile.write(errmessage)
    errfile.close()



                    
                

