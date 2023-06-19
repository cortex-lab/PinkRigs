function logapp = appendAndPrint(log, message, fid)
    %% Appends to log and prints message in terminal.
    %
    % Parameters:
    % -------------------
    % log: str
    %   Log.
    % message: str
    %   Message to add.
    % fid: open text file
    %   File to write to.
    %
    % Returns: 
    % -------------------
    % logapp: str
    %   Updated log.

    if nargin > 2 && ~isempty(fid)
        fprintf(fid,'%s\n',message);
    end
    logapp = append(log, message);
    fprintf('%s\n',message)
end