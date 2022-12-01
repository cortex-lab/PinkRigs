function logapp = appendAndPrint(log, message)
    logapp = append(log, message);
    fprintf('%s\n',message)
end