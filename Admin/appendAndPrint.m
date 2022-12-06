function logapp = appendAndPrint(log, message, fid)
    if nargin > 2 && ~isempty(fid)
        fprintf(fid,'%s\n',message);
    end
    logapp = append(log, message);
    fprintf('%s\n',message)
end