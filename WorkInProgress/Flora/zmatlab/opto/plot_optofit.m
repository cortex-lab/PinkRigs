%
function plot_optofit(glmData,plotParams,plotfit,fixedContrastPower)
plottype = plotParams.plottype; 
params2use = mean(glmData.prmFits,1);   
pHatCalculated = glmData.calculatepHat(params2use,'eval');
[grids.visValues, grids.audValues] = meshgrid(unique(glmData.evalPoints(:,1)),unique(glmData.evalPoints(:,2)));
[~, gridIdx] = ismember(glmData.evalPoints, [grids.visValues(:), grids.audValues(:)], 'rows');
plotData = grids.visValues;
plotData(gridIdx) = pHatCalculated(:,2);
if plotfit
    plotOpt.lineStyle = plotParams.LineStyle;
else
    plotOpt.lineStyle = 'none'; %plotParams.LineStyle;
end
plotOpt.Marker = 'none';
currBlock = glmData.dataBlock; 
%contrastPower = params.contrastPower{refIdx};

if strcmp(plottype, 'log') && exist('fixedContrastPower', 'var')
    contrastPower = fixedContrastPower;
    plotData = log10(plotData./(1-plotData));
elseif strcmp(plottype, 'log')
    tempFit = plts.behaviour.GLMmulti(currBlock, 'simpLogSplitVSplitA');
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

if plotfit
    plotOpt.lineStyle = 'none';
else
    plotOpt.lineStyle = plotParams.LineStyle;
end
plotOpt.Marker = plotParams.DotStyle;
plotOpt.MarkerSize = plotParams.MarkerSize; 
plotOpt.FaceAlpha = 0.1; 

if plotParams.addFake==1
    currBlock = addFakeTrials(currBlock); 
end 

visDiff = currBlock.stim_visDiff;
audDiff = currBlock.stim_audDiff;
responseDir = currBlock.response_direction;

% append one trialtype to each condition?


[visGrid, audGrid] = meshgrid(unique(visDiff),unique(audDiff));
maxContrast = max(abs(visGrid(1,:)));
fracRightTurns = arrayfun(@(x,y) mean(responseDir(ismember([visDiff,audDiff],[x,y],'rows'))==2), visGrid, audGrid);

visValues = abs(visGrid(1,:)).^contrastPower.*sign(visGrid(1,:))./(maxContrast.^contrastPower);
% instead of creating visValues you use both visGrid and audGrid and the
% parameters to compute f(s)....
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