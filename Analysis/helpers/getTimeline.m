function Timeline = getTimeline(varargin)
    %%% This function will load timeline.
    
    switch nargin
        case 1
            % expPath configuration
            expPath = varargin{1};
            [subject, expDate, expNum] = parseExpPath(expPath);
            load(fullfile(expPath, [expDate '_' expNum '_' subject '_Timeline.mat']),'Timeline');
        case 3
            % subject,expDate,expNum configuration
            subject = varargin{1};
            expDate = varargin{2};
            expNum = varargin{3};
            expPath = getExpPath(subject, expDate, expNum);
            load(fullfile(expPath, [expDate '_' expNum '_' subject '_Timeline.mat']),'Timeline');
        otherwise
            error('Wrong number of arguments.')
    end