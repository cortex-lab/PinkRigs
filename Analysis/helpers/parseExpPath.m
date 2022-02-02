function [subject, expDate, expNum, server] = parseExpPath(expPath)
    %%% Parses the expPath to retrieve server, subject, etc.
    
    splitStr = regexp(expPath,'\','split');
    server = ['\\' splitStr{3}];
    subject = splitStr{5};
    expDate = splitStr{6};
    expNum = str2double(splitStr{7});