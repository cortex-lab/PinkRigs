%% fit the glm on the opto mice 
% parameters to call are in the metadata so use the manual to read in the
% metadata

power = '30'; 
hemi = 'R';
%
params.subject = {['AV033']};
params.expDef = 'm';
params.checkEvents = 1;
exp2checkList = csv.queryExp(params);
isOptoData = logical(cellfun(@(x) numel(dir([x '\**\*optoMetaData.csv'])),exp2checkList.expFolder));
exp2checkList = exp2checkList(isOptoData,:); 
[exp2checkList.power,exp2checkList.inactivatedHemisphere] = cellfun(@(x) readOptoMeta(x),exp2checkList.expFolder);
exp2checkList = exp2checkList((cellfun(@(x) strcmp(x,power),exp2checkList.power)),:);

figure;
plt.behaviour.glmFit(exp2checkList,...
    'modelString','simpLog',...
    'useLaserTrials',0, ...
    'plotType','res',...
     'fitLineStyle','-',...
    'useCurrentAxes',1,...
     'datDotStyle','x');
%
exp2checkList = exp2checkList((cellfun(@(x) strcmp(x,hemi),exp2checkList.inactivatedHemisphere)),:);

%
plt.behaviour.glmFit(exp2checkList, ...
     'modelString','simpLog',...
    'useLaserTrials',1, ...
    'useCurrentAxes',1,...
    'fitLineStyle','-',...
     'plotType','res',...
    'datDotStyle','x');

%%
function [power,hemisphere] = readOptoMeta(expFolder)
d = dir([expFolder '\**\*optoMetaData.csv']);
d = csv.readTable([d.folder '/' d.name]); 
power = d.LaserPower_mW; 
hemisphere = d.Hemisphere; 
end %%