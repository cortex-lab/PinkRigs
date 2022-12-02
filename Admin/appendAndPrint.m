function logapp = appendAndPrint(log, message, fid)
    fprintf(fid,'%s\n',message);
    logapp = append(log, message);
    fprintf('%s\n',message)
end