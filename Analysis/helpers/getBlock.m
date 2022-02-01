function block = getBlock(varargin)
    %%% This function will load the block.
    
    switch nargin
        case 1
            % expPath configuration
            expPath = varargin{1};
            [subject, expDate, expNum] = parseExpPath(expPath);
            load(fullfile(expPath, [expDate '_' num2str(expNum) '_' subject '_block.mat']),'block');
        case 3
            % subject,expDate,expNum configuration
            subject = varargin{1};
            expDate = varargin{2};
            expNum = varargin{3};
            expPath = getExpPath(subject, expDate, expNum);
            load(fullfile(expPath, [expDate '_' num2str(expNum) '_' subject '_block.mat']),'block');
        otherwise
            error('Wrong number of arguments.')
    end