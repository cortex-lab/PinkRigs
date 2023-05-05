clear all;,
params.subject  = {['AV036'];['AV038'];['AV033'];['AV031'];['AV029']};
%params.subject  = {['AV038']};
params.expDef = 'm'; 
params.checkEvents = '1'; 
params.expDate = {['2022-04-24:2023-04-28']}; 
exp2checkList = csv.queryExp(params);
params = csv.inputValidation(exp2checkList);
extracted = getOptoData(exp2checkList, 'reverse_opto', 1,'combMice',0,'combHemispheres',0,'combDates',1,'combPowers',1); 

% plot the control vs the opto on the same plot for each 'extracted'


%%
% fit and plot each set of data
%
% fit sets that determine which parameters or combinations of parameters
% are allowed to change from fitting control trials to fitting opto trials
opto_fit_sets = logical([
    [0,0,0,0,0,0]; ... 
    [1,1,1,1,1,1]; ...
    [1,0,0,0,0,0]; ...
    [0,1,0,0,0,0]; ...
    [0,0,1,0,0,0]; ...
    [0,0,0,0,1,0]; ...
    [0,0,0,0,0,1]; ...
    [0,1,1,1,1,1]; ...   
    [1,0,1,1,1,1]; ...    
    [1,1,0,1,1,1]; ...    
    [1,1,1,1,0,1]; ...    
    [1,1,1,1,1,0]; ...    
]);

plot_model_pred = zeros(size(opto_fit_sets,1),1); % indices of models to plot
plot_model_pred(2) = 1; 
shouldPlot = 1; 
plotfit = 1; % whether to connect the data or plot actual fits
plotParams.plottype = 'log'; 
for s=1:numel(extracted.data)    
    currBlock = extracted.data{s};
    nTrials(s) = numel(currBlock.is_blankTrial); 
    keepIdx = currBlock.response_direction & currBlock.is_validTrial & currBlock.is_laserTrial & abs(currBlock.stim_audAzimuth)~=30;
    optoBlock = filterStructRows(currBlock, keepIdx); 
    keepIdx = currBlock.response_direction & currBlock.is_validTrial & ~currBlock.is_laserTrial & abs(currBlock.stim_audAzimuth)~=30;
    controlBlock = filterStructRows(currBlock, keepIdx); 
    
    % fit and plot
    controlfit = plts.behaviour.GLMmulti(controlBlock, 'simpLogSplitVSplitA');
    controlfit.fit; 
    
    if shouldPlot
        figure; 
        plotParams.LineStyle = '-';
        plotParams.DotStyle = ['.'];
        plotParams.MarkerSize = 24; 
        plot_optofit(controlfit,plotParams,plotfit)
        hold on; 
        title(sprintf('%s,%.0d opto,%.0d control trials,%.0f mW, %.0f', ...
            extracted.subject{s},...
            numel(optoBlock.is_blankTrial),... 
            numel(controlBlock.is_blankTrial),... 
            extracted.optoParams{s}(1),...
            extracted.optoParams{s}(2)))
    end


    %
    for model_idx = 1:size(opto_fit_sets,1)
        optoBlock.freeP  = opto_fit_sets(model_idx,:);
        orifit = plts.behaviour.GLMmulti(optoBlock, 'simpLogSplitVSplitA');
        orifit.prmInit = controlfit.prmFits;
        orifit.fit; 
        opto_fit_logLik(s,model_idx) = orifit.logLik;
        % how the parameters actually change 
        opto_fit_params(s,model_idx,:) = orifit.prmFits;

        if shouldPlot && plot_model_pred(model_idx)
           plotParams.LineStyle = '--';
           plotParams.DotStyle = 'o';
           plotParams.MarkerSize = 8; 
           plot_optofit(orifit,plotParams,plotfit)
        end

    end
    %
end

%%
% summary plots for cross-validation 
% only include things that are more than 2k trials
paramLabels = categorical({'bias','Vipsi','Vcontra','Aipsi','Acontra'}); 
opto_fit_logLik_ = opto_fit_logLik(nTrials>2000,:); 

% normalise the log2likelihood

 
% plot order; 
% calculate fit improvement by varying each predictor 
best_deltaR2 = opto_fit_logLik_(:,1) - opto_fit_logLik_(:,2);
deltaR2 = (opto_fit_logLik_(:,1)-opto_fit_logLik_(:,3:7))./best_deltaR2;
%

cvR2 = (opto_fit_logLik_(:,2)-opto_fit_logLik_(:,8:12))./best_deltaR2;

%% individual plots 
figure; 
plot(deltaR2');
figure;plot(cvR2'); 

%% plot bar plot of summary 
figure;
errorbar(paramLabels,median(deltaR2),zeros(size(deltaR2,2),1),mad(deltaR2),'black',"LineStyle","none");
hold on; 
bar(paramLabels,median(deltaR2),'black');
hold on; 
errorbar(paramLabels,median(cvR2),mad(cvR2),zeros(size(deltaR2,2),1),'green',"LineStyle","none");
hold on;
bar(paramLabels,median(cvR2),'green'); 

%% 
% summary plot for how the bias term changes between controlfit and bias
% fit

figure; 
ptype = 1; 
plot(opto_fit_params(nTrials>2000,1,ptype),opto_fit_params(nTrials>2000,2,ptype),'o')
hold on; 
plot([-5,5],[-5,5],'k--')
xlabel('bias,control fit')
ylabel('bias,full fit')
ylim([-5,5])
%%
function plot_optofit(glmData,plotParams,plotfit)
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

if strcmp(plottype, 'log')
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