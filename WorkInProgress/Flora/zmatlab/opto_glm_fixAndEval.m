% glm fit for the opto data round #2
% 
subject_list = {'AV029';'AV033';'AV033';'AV031';'AV031'}; 
hemis = ['R','R','L','L','L'];
powers = {'30';'30';'30';'30';'15'};

% subject_list = {'AV033'}; 
% hemis = ['R'];
% powers = {'30'};


for s=1:numel(subject_list)
    clear params
    params.subject = {[subject_list{s}]};
    hemi = hemis(s);
    power = powers{s}; 
    params.expDef = 'm';
    params.checkEvents = 1;
    exp2checkList = csv.queryExp(params);
    isOptoData = logical(cellfun(@(x) numel(dir([x '\**\*optoMetaData.csv'])),exp2checkList.expFolder));
    exp2checkList = exp2checkList(isOptoData,:); 
    [exp2checkList.power,exp2checkList.inactivatedHemisphere] = cellfun(@(x) readOptoMeta(x),exp2checkList.expFolder);

    
    figure;
    controlfit = plts.behaviour.glmFit(exp2checkList,...
        'modelString','simpLogSplitVSplitA',...
        'useLaserTrials',0, ...
        'plotType','res',...
         'fitLineStyle','--',...
        'useCurrentAxes',1,...
         'datDotStyle','None');
    
    
    %
    exp2checkList = exp2checkList((cellfun(@(x) strcmp(x,power),exp2checkList.power)),:);
    exp2checkList = exp2checkList((cellfun(@(x) strcmp(x,hemi),exp2checkList.inactivatedHemisphere)),:); 


    controlfit = controlfit{1,1};
    ctrl_fit(s) = controlfit.logLik; 
    %
    
    params = csv.inputValidation(exp2checkList);
    extracted = plts.behaviour.getTrainingData(exp2checkList);
    currBlock = extracted.data{1};
    keepIdx = currBlock.response_direction & currBlock.is_validTrial & currBlock.is_laserTrial & abs(currBlock.stim_audAzimuth)~=30;
    optoBlock = filterStructRows(currBlock, keepIdx); 

    
    optoBlock.freeP  = logical([0,0,0,0,0,0]); 
    orifit = plts.behaviour.GLMmulti(optoBlock, 'simpLogSplitVSplitA');
    orifit.prmInit = controlfit.prmFits;
    orifit.fit; 
    figure; plot_optofit(orifit,'res');
    nochange_fit(s) = orifit.logLik;
    

    optoBlock.freeP  = logical([1,0,0,0,0,0]); 
    optofit = plts.behaviour.GLMmulti(optoBlock, 'simpLogSplitVSplitA');
    optofit.prmInit = controlfit.prmFits;
    optofit.fit; 
    figure; plot_optofit(optofit,'res');
    bias_fit(s) = optofit.logLik;
    
    optoBlock.freeP  = logical([1,1,1,1,1,1]);
    optofit_additive = plts.behaviour.GLMmulti(optoBlock, 'simpLogSplitVSplitA'); 
    optofit_additive.prmInit = controlfit.prmFits;
    optofit_additive.fit;
    plot_optofit(optofit_additive,'res');
    additive_fit(s) = optofit_additive.logLik;


    uninitialised_additive = plts.behaviour.glmFit(exp2checkList,...
    'modelString','simpLogSplitVSplitA',...
    'useLaserTrials',1, ...
    'plotType','res',...
     'fitLineStyle','--',...
    'useCurrentAxes',1,...
     'datDotStyle','o');

    uninitialised_additive_fit(s) = uninitialised_additive{1,1}.logLik;
    
    %
end
%%
figure; 
plot(bias_fit,additive_fit,'r.','MarkerSize',24)
hold on
plot(nochange_fit,additive_fit,'b.','MarkerSize',24)

hold on 
plot([0,2],[0,2],'k--')
xlabel('bias')
for i=1:numel(subject_list)
    x = [bias_fit(i) nochange_fit(i)]; 
    y = [additive_fit(i) additive_fit(i)]; 
    plot(x,y,'b'); 
end
ylabel('all')
%newfit.fit; 

%%
fits = [nochange_fit./additive_fit;bias_fit./additive_fit];
figure; 
hold all;
for i=1:numel(subject_list)
    plot(fits(:,i));
end
%plot(mean(fits))

%%
function [power,hemisphere] = readOptoMeta(expFolder)
d = dir([expFolder '\**\*optoMetaData.csv']);
d = csv.readTable([d.folder '/' d.name]); 
power = d.LaserPower_mW; 
hemisphere = d.Hemisphere; 
end %%


function plot_optofit(glmData,plottype)
params2use = mean(glmData.prmFits,1);   
pHatCalculated = glmData.calculatepHat(params2use,'eval');
[grids.visValues, grids.audValues] = meshgrid(unique(glmData.evalPoints(:,1)),unique(glmData.evalPoints(:,2)));
[~, gridIdx] = ismember(glmData.evalPoints, [grids.visValues(:), grids.audValues(:)], 'rows');
plotData = grids.visValues;
plotData(gridIdx) = pHatCalculated(:,2);
plotOpt.lineStyle = '-';
plotOpt.Marker = 'none';
currBlock = glmData.dataBlock; 
%contrastPower = params.contrastPower{refIdx};

if strcmp(plottype, 'log')
    tempFit = plt.behaviour.GLMmulti(currBlock, 'simpLogSplitVSplitA');
    tempFit.fit;
    tempParams = mean(tempFit.prmFits,1);
    contrastPower  = tempParams(strcmp(tempFit.prmLabels, 'N'));
    plotData = log10(plotData./(1-plotData));
else
    contrastPower= 1;
end

visValues = (abs(grids.visValues(1,:))).^contrastPower.*sign(grids.visValues(1,:));
lineColors = plts.general.selectRedBlueColors(grids.audValues(:,1));
plts.general.rowsOfGrid(visValues, plotData, lineColors, plotOpt);

plotOpt.lineStyle = 'none';
plotOpt.Marker = 'x';


visDiff = currBlock.stim_visDiff;
audDiff = currBlock.stim_audDiff;
responseDir = currBlock.response_direction;
[visGrid, audGrid] = meshgrid(unique(visDiff),unique(audDiff));
maxContrast = max(abs(visGrid(1,:)));
fracRightTurns = arrayfun(@(x,y) mean(responseDir(ismember([visDiff,audDiff],[x,y],'rows'))==2), visGrid, audGrid);

visValues = abs(visGrid(1,:)).^contrastPower.*sign(visGrid(1,:))./(maxContrast.^contrastPower);
if strcmp(plottype, 'log')
    fracRightTurns = log10(fracRightTurns./(1-fracRightTurns));
end
plts.general.rowsOfGrid(visValues, fracRightTurns, lineColors, plotOpt);

xlim([-1 1])
midPoint = 0.5;
xTickLoc = (-1):(1/8):1;
if strcmp(plottype, 'log')
    ylim([-2.6 2.6])
    midPoint = 0;
    xTickLoc = sign(xTickLoc).*abs(xTickLoc).^contrastPower;
end

box off;
xTickLabel = num2cell(round(((-maxContrast):(maxContrast/8):maxContrast)*100));
xTickLabel(2:2:end) = deal({[]});
set(gca, 'xTick', xTickLoc, 'xTickLabel', xTickLabel);

%title(sprintf('%s: %d Tri in %s', extracted.subject{i}, length(responseDir), extracted.blkDates{i}{1}))
xL = xlim; hold on; plot(xL,[midPoint midPoint], '--k', 'linewidth', 1.5);
yL = ylim; hold on; plot([0 0], yL, '--k', 'linewidth', 1.5);
end