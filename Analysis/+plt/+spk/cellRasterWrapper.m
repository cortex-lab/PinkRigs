function cellRasterWrapper(varargin)
varargin = ['expDef', {{{'t'; 'p'}}}, varargin];
varargin = ['paramTag', 'default', varargin];
params = csv.inputValidation(varargin{:});
expList = csv.queryExp(params);

if size(expList) > 2
    error('You have requested > 2 sessions--Not possible')
end
loadedData = csv.loadData(expList, 'dataType', {'events';'probe'}, 'object', {'all'; 'spikes'});
for i = 1:height(loadedData)
    if ~containsloadedData.expDef{i}
    end
    [eventTimes, trialGroups, opt] = plt.spk.rasterParams(loadedData.dataEvents{i});
end

plt.spk.cellRasterNew(ev, spk);
end