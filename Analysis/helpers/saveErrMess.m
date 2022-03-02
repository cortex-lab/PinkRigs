function saveErrMess(errorMessage,errorPath)
    %%% This function will save in the given path a json containing the
    %%% error message.
    
    errorMsge = jsonencode(errorMessage);
    fid = fopen(errorPath, 'w');
    fprintf(fid, '%s', errorMsge);
    fclose(fid);