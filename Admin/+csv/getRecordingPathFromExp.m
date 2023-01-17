function recList = getRecordingPathFromExp(varargin)
    %% Fetches the recording location for a list of experiments.
    %
    % Parameters (optional):
    % -------------------
    % Classic PinkRigs inputs.
    % KSversion: str
    %   Version of kilosort to look at (usually PyKS).
    %
    % Returns: 
    % -------------------
    % recList: cell array
    %   List of kilosort directories.

    varargin = ['KSversion', 'PyKS', varargin];
    varargin = [varargin, 'checkAlignEphys', true]; % force it
    params = csv.inputValidation(varargin{:});
    exp2checkList = csv.queryExp(params);
    
    %% Find all the relevant ephys files#
    cc = 1;
    for ee = 1:size(exp2checkList,1)
    
        % Assign variables from exp2checkList to ease of use later
        expDate = exp2checkList.expDate{ee,1};
        expNum = exp2checkList.expNum{ee,1};
        subject = exp2checkList.subject{ee,1};
        expFolder = exp2checkList.expFolder{ee,1};
        KSversion = exp2checkList.KSversion{ee,1};
    
        % Get the alignment file
        pathStub = fullfile(expFolder, [expDate '_' expNum '_' subject]);
        alignmentFile = [pathStub '_alignment.mat'];
    
        alignment = load(alignmentFile, 'ephys');

        if ~isempty(fieldnames(alignment))
            if ~strcmp(alignment.ephys,'error')
                for probeNum = 1:numel(alignment.ephys)
                    switch KSversion
                        case 'KS2'
                            KSFolder{cc} = fullfile(alignment.ephys(probeNum).ephysPath,'kilosort2');
                        case 'PyKS'
                            KSFolder{cc} = fullfile(alignment.ephys(probeNum).ephysPath,'PyKS','output');
                    end

                    cc = cc + 1;
                end
            end
        end
    end

    recList = unique(KSFolder);