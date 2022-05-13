function loadedData = loadData(varargin)
%% Load ev and/or spk data from particular mice and/or dates
params = inputValidation(varargin{:});
params = csv.addDefaultParam(params, 'loadTag', 'ev');
extractedExps = csv.queryExp(params);

for i=1:height(extractedExps)
    currExp = extractedExps(i,:);
    currMouseIdx = strcmpi(params.subject, currExp.subject);
    currLoadTag = params.loadTag{currMouseIdx};
    pathStub = cell2mat([currExp.expDate '_' currExp.expNum '_' currExp.subject]);

    clear ev spk blk;

    %Load ev data if requested
    if contains(currLoadTag, {'ev'})
        preProcPath = cell2mat([currExp.expFolder '\' pathStub '_preprocData.mat']);
        if ~exist(preProcPath, 'file'); continue; end
        ev = load(preProcPath, 'ev');
        if ~exist('ev', 'var'); continue; end
        loadedData.ev{i,1} = ev.ev;
    end

    %Load spk data if requested
    if contains(currLoadTag, {'ev'})
        preProcPath = cell2mat([currExp.expFolder '\' pathStub '_preprocData.mat']);
        if ~exist(preProcPath, 'file')
            loadedData.ev{i,1} = nan;
            continue;
        else
            ev = load(preProcPath, 'ev');
            loadedData.ev{i,1} = ev.ev;
        end
    end
end

