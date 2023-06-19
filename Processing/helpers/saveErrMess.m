function saveErrMess(errorMessage,errorPath)
    %% Writes a .json file with a message.
    %
    % Parameters:
    % -------------------
    % errorMessage: str
    %   Message to write
    % errorPath: str
    %   Path
    
    errorMsge = jsonencode(errorMessage);
    fid = fopen(errorPath, 'w');
    fprintf(fid, '%s', errorMsge);
    fclose(fid);