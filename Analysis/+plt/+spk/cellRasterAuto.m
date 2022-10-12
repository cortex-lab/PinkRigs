function cellRasterAuto(varargin)
varargin = ['expDef', {{{'t'; 'p'}}}, varargin];
varargin = ['paramTag', 'default', varargin];
varargin = ['checkSpikes', '1', varargin];
params = csv.inputValidation(varargin{:});

%If the evTimes and triGrps are not provided, load them.
fprintf('*No evTimes provided, so generating them based on inputs...\n')
expList = csv.queryExp(params);

dataTypes = {'events';'probe';'probe'};
objects = {'all';'spikes';'clusters'};
attrtributes = {{'all'};{'times', 'clusters', 'amps'};{'depths', '_av_xpos', '_av_IDs'}};

loadedData = csv.loadData(expList, 'dataType', dataTypes, 'object', objects,...
    'attribute', attrtributes);

uniDates = unique(expList.expDate);
dat = cell(length(uniDates),1);
for i = 1:length(uniDates)
    currDate = uniDates{i};
    currData = loadedData(contains(loadedData.expDate,currDate),:);

    expDefsUsed = cellfun(@preproc.getExpDefRef,currData.expDef,'uni',0);
    if length(unique(expDefsUsed)) > 2
        fprintf('WARNING: 3 expdefs for %s. Will skip... \n', currDate);
        continue
    elseif length(unique(expDefsUsed)) < height(currData)
        fprintf('WARNING: redundant expdefs for %s. Skipping smallest... \n', currDate);
        keepIdx = ones(height(currData),1)>0;
        for expName = unique(expDefsUsed)'
            expIdx = strcmpi(expName, expDefsUsed);
            if sum(expIdx) == 1; continue; end
            durations = str2double(currData.expDuration);
            keepIdx(expIdx) = durations(expIdx)~=max(durations(expIdx));
        end
        currData = currData(keepIdx,:);
    end
    if height(currData) > 2
        error('Should not be more than 2 experiments by this stage...')
    end

    funcStub = 'plt.spk.rasterParams.';
    if strcmpi(currData.paramTag{1}, 'default')
        if any(contains(expDefsUsed, {'Training', 'Passive'}))
            paramFunc = str2func([funcStub 'defaultActivePassive']);
            dat{i,1} = paramFunc(currData);
        end
    else
        dat{i,1} = paramFunc(currData);
    end
end
plt.spk.cellRaster(dat)
