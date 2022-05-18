function expList = loadData(varargin)
%% Load ev and/or spk data from particular mice and/or dates
params = csv.inputValidation(varargin{:});
params = csv.addDefaultParam(params, 'loadTag', 'ev');
expList = csv.queryExp(params);

newFields = {'blkData'; 'evData'; 'spkData'};
for i = 1:length(newFields)
expList.(newFields{i}) = cell(size(expList,1),1);
end

if isempty(expList); error('No sessions matched params'); end
for i=1:height(expList)
    clear ev spk blk;
    currExp = expList(i,:);
    currMouseIdx = strcmpi(params.subject, currExp.subject);
    currLoadTag = params.loadTag{currMouseIdx};
    pathStub = cell2mat([currExp.expDate '_' currExp.expNum '_' currExp.subject]);
    
    %Load ev/spk data if requested
    if contains(currLoadTag, {'ev', 'spk'})
        preProcPath = cell2mat([currExp.expFolder '\' pathStub '_preprocData.mat']);
        if ~exist(preProcPath, 'file'); continue; end
        if contains(currLoadTag, {'ev'})
            ev = load(preProcPath, 'ev');
            if exist('ev', 'var')
                expList.evData{i} = ev.ev;
            end
        end
        if contains(currLoadTag, {'spk'})
            spk = load(preProcPath, 'spk');
            if exist('spk', 'var')
                expList.spkData{i} = spk.spk;
            end
        end    
    end


    %Load block data if requested
    if contains(currLoadTag, {'blk'})
        preProcPath = cell2mat([currExp.expFolder '\' pathStub '_block.mat']);
        if ~exist(preProcPath, 'file'); continue; end
        blk = load(preProcPath, 'block');
        if exist('blk', 'var')
            expList.blkData{i} = blk.block;
        end
    end
end

expList = removevars(expList,{'expDuration'; 'block'; 'timeline';...
    'frontCam'; 'sideCam'; 'eyeCam'; 'micDat'; 'ephysFolderExists'; ...
    'alignBlkFrontSideEyeMicEphys'; 'faceMapFrontSideEye'; 'issorted'; ...
    'preProcSpkEV'; 'expFolder'});

for i = 1:length(newFields)
    emptyCells = cellfun(@isempty, expList.(newFields{i}));
    if all(emptyCells)
        expList = removevars(expList, newFields{i});
    else
        expList.(newFields{i})(emptyCells) = {nan};
    end
end
end
