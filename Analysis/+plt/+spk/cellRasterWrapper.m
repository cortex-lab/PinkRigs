function cellRasterWrapper(varargin)
varargin = ['expDef', {{{'t'; 'p'}}}, varargin];
varargin = ['paramTag', 'default', varargin];
params = csv.inputValidation(varargin{:});
expList = csv.queryExp(params);

if size(expList) > 2
    error('You have requested > 2 sessions--Not possible')
end
loadedData = csv.loadData(expList, 'loadTag', 'evspk');
for i = 1:height(loadedData)
    if loadedData.expD
    end
    [eventTimes, trialGroups, opt] = paramFunc(loadedData.evData{i});
end

plt.spk.cellRasterNew(ev, spk);
end
