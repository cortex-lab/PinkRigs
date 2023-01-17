function errorMessages = fetchErrorMessage(varargin)
    %%% This function will return all the error messages for a (possibly
    %%% specified) list of experiments. You can also specify which error
    %%% messages you want to look at with argument "whichMessage", which
    %%% should be cell containing the fields of the mice csv you want to
    %%% check, e.g.:
    %%%     whichMessage = {'align','extractSpikes'}
    %%% 'align' will fetch error messages for all alignments, and
    %%% 'extractSpikes' for the spikes extraction.
    %%% A few general options are: 'all', 'align', 'aligncam', 'fmapcam'

    varargin = ['videoNames', {{{'frontCam';'sideCam';'eyeCam'}}}, varargin];
    varargin = ['whichMessage', 'all', varargin];
    params = csv.inputValidation(varargin{:});
    exp2checkList = csv.queryExp(params);

    %% Loop through all experiments to fetch error messages

    errorMessages = struct();
    for ee = 1:size(exp2checkList,1)
        % Get expInfo for current experiment (passed to sub functions)
        expInfo = exp2checkList(ee,:);

        % Assign variables from exp2checkList to ease of use later
        whichMessage = exp2checkList.whichMessage{ee, 1};
        videoNames = exp2checkList.videoNames{ee, 1};

        % Get fields to check
        fieldsToCheck = contains(lower(expInfo.Properties.VariableNames),lower(whichMessage));
        % Add specific cases
        if any(strcmpi(whichMessage,'all'))
            fieldsToCheck = fieldsToCheck | contains(lower(expInfo.Properties.VariableNames), {'align','fMap','issorted','extract'});
        end
        if any(strcmpi(whichMessage,'align'))
            fieldsToCheck = fieldsToCheck | contains(lower(expInfo.Properties.VariableNames), {'align'});
        end
        if any(strcmpi(whichMessage,'aligncam'))
            fieldsToCheck = fieldsToCheck | contains(lower(expInfo.Properties.VariableNames), {'align'}) & ...
                contains(lower(expInfo.Properties.VariableNames), {'cam'});
        end
        if any(strcmpi(whichMessage,'fmapcam'))
            fieldsToCheck = fieldsToCheck | contains(lower(expInfo.Properties.VariableNames), {'fmap'}) & ...
                contains(lower(expInfo.Properties.VariableNames), {'cam'});
        end
        fieldsToCheckNames = expInfo.Properties.VariableNames(fieldsToCheck);

        % Get error status. Go through all fields otherwise crashes.
        erroredFields = struct();
        for i = expInfo.Properties.VariableNames
            erroredFields.(lower(i{1})) = expInfo.(i{1});
        end

        % Anonymous function to decide whether a message should be fetched
        shouldFetch = @(x) (any(contains(fieldsToCheckNames,{'all';x}, 'IgnoreCase',true))...
            & contains(erroredFields.(lower(x)){1}, '2'));

        % Extract error messages
        % Align ephys
        if shouldFetch('alignEphys')
            errorMessageFile = dir(fullfile(expInfo.expFolder{1},'AlignEphysError.json'));
            errorMessages(ee).alignEphys = readJSON(errorMessageFile);
        end

        % Align block
        if shouldFetch('alignBlock')
            errorMessageFile = dir(fullfile(expInfo.expFolder{1},'AlignBlockError.json'));
            errorMessages(ee).alignBlock = readJSON(errorMessageFile);
        end

        % Align videos
        for v = 1:numel(videoNames)
            if shouldFetch(['align' videoNames{v}])
                errorMessageFile = dir(fullfile(expInfo.expFolder{1},'ONE_preproc',videoNames{v},sprintf('AlignVideoError_%s.json',videoNames{v})));
                alignNameCam = fieldsToCheckNames(strcmpi(fieldsToCheckNames,['align' videoNames{v}]));
                errorMessages(ee).(alignNameCam{1}) = readJSON(errorMessageFile);
            end
        end

        % Align mic
        if shouldFetch(['align' videoNames{v}])
            errorMessageFile = dir(fullfile(expInfo.expFolder{1},'ONE_preproc','mic','AlignMicError.json'));
            errorMessages(ee).alignMic = readJSON(errorMessageFile);
        end

        % Facemap
        %%% NO ERROR MESSAGE?

        % Sorting 
        if shouldFetch('issortedPyKS')
            % Happens in the associated ephys folder
            alignmentFile = dir(fullfile(expInfo.expFolder{1},'*_alignment.mat'));
            if ~isempty(alignmentFile)
                alignment = load(fullfile(alignmentFile.folder,alignmentFile.name), 'ephys');
                for probeNum = 1:numel(alignment.ephys)
                    probeRef = sprintf('probe%s',num2str(probeNum-1));
                    errorMessageFile = dir(fullfile(alignment.ephys(probeNum).ephysPath,'pyKS','pyKS_error.json'));
                    errorMessages(ee).issortedPyKS.(probeRef) = readJSON(errorMessageFile);
                end
            end
        end

        % Extract events
        if shouldFetch('extractEvents')
            errorMessageFile = dir(fullfile(expInfo.expFolder{1},'ONE_preproc','events','AlignBlockError.json'));
            errorMessages(ee).extractEvents = readJSON(errorMessageFile);
        end

        % Extract spikes
        if shouldFetch('extractSpikes')
            % Happens in the associated ephys folder
            alignmentFile = dir(fullfile(expInfo.expFolder{1},'*_alignment.mat'));
            if ~isempty(alignmentFile)
                alignment = load(fullfile(alignmentFile.folder,alignmentFile.name), 'ephys');
                for probeNum = 1:numel(alignment.ephys)
                    probeRef = sprintf('probe%s',num2str(probeNum-1));
                    errorMessageFile = dir(fullfile(expInfo.expFolder{1},'ONE_preproc',probeRef,'GetSpkError.json'));
                    errorMessages(ee).extractSpikes.(probeRef) = readJSON(errorMessageFile);
                end
            end
        end
    end

    % Fill up the whole structure artificially.
    fnames = fieldnames(errorMessages);
    errorMessages(size(exp2checkList,1)+1).(fnames{1}) = [];
    errorMessages(size(exp2checkList,1)+1) = [];
end 

function errText = readJSON(errFile)
    if ~isempty(errFile)
        fid = fopen(fullfile(errFile.folder,errFile.name));
        errText = jsondecode(char(fread(fid, inf)'));
        fclose(fid);
    else
        errText = 'No error message found.';
    end
end