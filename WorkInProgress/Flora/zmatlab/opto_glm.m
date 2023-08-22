clc; clear all;
extracted = loadOptoData('balanceTrials',0,'sepMice',1,'reExtract',1,'sepHemispheres',1); 



%%
% fit and plot each set of data
%
% fit sets that determine which parameters or combinations of parameters
% are allowed to change from fitting control trials to fitting opto trials
opto_fit_sets = logical([
    [0,0,0,0,0,0]; ... 
    [1,1,1,0,1,1]; ...
    [1,0,0,0,0,0]; ...
    [1,1,0,0,0,0]; ... 
    [1,0,1,0,0,0]; ...
    [1,0,0,0,1,0]; ...
    [1,0,0,0,0,1]; ...
    [0,1,0,0,0,0]; ... 
    [0,0,1,0,0,0]; ...
    [0,0,0,0,1,0]; ...
    [0,0,0,0,0,1]; ...
    [0,1,1,0,1,1]; ...   
    [1,0,1,0,1,1]; ...    
    [1,1,0,0,1,1]; ...    
    [1,1,1,0,0,1]; ...    
    [1,1,1,0,1,0]; ...    
]);

%%

plot_model_pred = zeros(size(opto_fit_sets,1),1); % indices of models to plot
plot_model_pred(2) = 1; 
shouldPlot = 1; 

plotfit = 1; % whether to connect the data or plot actual fits
plotParams.plottype = 'sigmoid'; 
for s=1:numel(extracted.data)    
    currBlock = extracted.data{s};
    nTrials(s) = numel(currBlock.is_blankTrial); 
    optoBlock = filterStructRows(currBlock, currBlock.is_laserTrial); 
    controlBlock = filterStructRows(currBlock, ~currBlock.is_laserTrial);

    % 
    %optoBlock = addFakeTrials(optoBlock);
    %controlBlock = addFakeTrials(controlBlock);

    % fit and plot
    controlfit = plts.behaviour.GLMmulti(controlBlock, 'simpLogSplitVSplitA');
    controlfit.fit; 
    control_fit_params(s,:)= controlfit.prmFits; 
    
    if shouldPlot
        figure; 
        plotParams.LineStyle = '-';
        plotParams.DotStyle = ['.'];
        plotParams.MarkerSize = 24; 
        plot_optofit(controlfit,plotParams,plotfit)
        hold on; 
        title(sprintf('%s,%.0d opto,%.0d control trials, %.0f mW, %.0f', ...
            extracted.subject{s},...
            numel(optoBlock.is_blankTrial),... 
            numel(controlBlock.is_blankTrial),...
            extracted.power{s},...
            extracted.hemisphere{s}))
    end


    %
    for model_idx = 1:size(opto_fit_sets,1)
        optoBlock.freeP  = opto_fit_sets(model_idx,:);
        orifit = plts.behaviour.GLMmulti(optoBlock, 'simpLogSplitVSplitA');
        orifit.prmInit = controlfit.prmFits;
        orifit.fitCV(5); 
        opto_fit_logLik(s,model_idx) = mean(orifit.logLik);
        % how the parameters actually change 
        opto_fit_params(s,model_idx,:) = mean(orifit.prmFits,1);

        if shouldPlot && plot_model_pred(model_idx)
           %

% figure;  orifit.prmFits(4)
    	   %orifit.prmFits(4) = controlfit.prmFits(4);
           plotParams.LineStyle = '--';
           plotParams.DotStyle = 'o';
           plotParams.MarkerSize = 8; 
           plot_optofit(orifit,plotParams,plotfit,orifit.prmInit(4))
        end

    end
    %
end

%%
% summary plots for cross-validation 
% only include things that are more than 2k trials
paramLabels = categorical({'bias','Vipsi','Vcontra','Aipsi','Acontra'}); 

% normalise the log2likelihood


 

% plot order; % calculate fit improvement by varying each predictor 

best_deltaR2 = opto_fit_logLik(:,1) - opto_fit_logLik(:,2);
deltaR2 = (opto_fit_logLik(:,1)-opto_fit_logLik(:,3:7))./best_deltaR2;


cvR2 = (opto_fit_logLik(:,2)-opto_fit_logLik(:,8:12))./best_deltaR2;

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
% compare the actual values for each mice 

figure; 

bar([1,2],median(opto_fit_logLik(:,2:3)),['black']);
hold on
for m=1:size(opto_fit_logLik,1)
    plot([1,2],[opto_fit_logLik(m,2),opto_fit_logLik(m,3)])
    hold on 
end 
[h,p]= ttest(opto_fit_logLik(:,2),opto_fit_logLik(:,3));

%%
labels = categorical({'bias','Vipsi','Vcontra','Aipsi','Acontra'}); 

figure;
bar([1,2,3,4,5],median(opto_fit_logLik(:,3:7)),['black']);
hold on
for m=1:size(opto_fit_logLik,1)
    plot([1,2,3,4,5],[opto_fit_logLik(m,3:7)])
    hold on 
end 



%%

%% 
% summary plot for how the each term changes between controlfit and full
% refit
% fit

figure; 
for ptype=1:numel(paramLabels)
    subplot(1,numel(paramLabels),ptype)
    plot(opto_fit_params(:,1,ptype),opto_fit_params(:,1,ptype)+opto_fit_params(:,2,ptype),'o')
    hold on; 
    plot([-5,5],[-5,5],'k--')
    xlabel(sprintf('%s,control fit',paramLabels(ptype)))
    ylabel(sprintf('%s,full fit',paramLabels(ptype)))
    ylim([-5,5])
end 
%%
function plot_optofit(glmData,plotParams,plotfit, fixedContrastPower)
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