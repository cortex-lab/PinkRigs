function [subject, expDate, expNum, server] = parseExpPath(expPath)
    %% Parses the expPath
    %
    % Parameters:
    % -------------------
    % expPath: str
    %   Path to the experiment
    %
    % Returns: 
    % -------------------
    % subject: str
    %   Name of the subject
    % expDate: str
    %   Date of the experiment
    % expNum: str
    %   Number of the experiment
    % server: str
    %   Server on which data is stored

    splitStr = regexp(expPath,'\','split');
    server = ['\\' splitStr{3}];
    subject = splitStr{5};
    expDate = splitStr{6};
    expNum = splitStr{7};