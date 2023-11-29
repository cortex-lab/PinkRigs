clc; clear all;
extracted = loadOptoData('balanceTrials',0,'sepMice',1,'reExtract',1,'sepHemispheres',1,'sepDiffPowers',1); 

% prepare params so that they can be matched
powers = [extracted.power{:}]; 
powers([extracted.hemisphere{:}]==0) = powers([extracted.hemisphere{:}]==0)/2; 
hemispheres = [extracted.hemisphere{:}]; 
power_set = [10];
subjects = [extracted.subject{:}]; 
unique_subjects = unique(subjects);


powerSubjectComb  = combvec(1:numel(unique_subjects),power_set); 
%%
% %
for s=1:size(powerSubjectComb,2)
    subject = unique_subjects(powerSubjectComb(1,s)); 
    p = powerSubjectComb(2,s);
    % get left, right and bilateral performance
    dL = get_delta_choice(extracted.data{(subjects==subject) & (hemispheres==-1) & (powers==p)});
    dR = get_delta_choice(extracted.data{(subjects==subject) & (hemispheres==1) & (powers==p)});

    biBlock = extracted.data{(subjects==subject) & (hemispheres==0) & (powers==p)};
   

    % plot ctrl+deltaL+deltaR vs actual bilateral
    plot_additive_pred(biBlock,(dL+dR)); 
end 

%%
% do the modelling,i.e. 


% % sum the delta bias for each uni 
% fit control for bi
% fit opto with control+delta bias 
% refit opto with full 
% assess logLik
freeP_unilateral = logical([1,0,0,0,0,0]);
%freeP_unilateral = logical([1,1,1,0,1,1]);

should_plot = 1; plotfit = 1; % whether to connect the data or plot actual fits
plotParams.plottype = 'log'; 

for s=1:size(powerSubjectComb,2)
    subject = unique_subjects(powerSubjectComb(1,s)); 
    p = powerSubjectComb(2,s);
    % fit unilateral inhibition, ctrl vs opto, allowing the bias only to change
    [paramsL,dParamsL] = get_delta_fitted(extracted.data{(subjects==subject) & (hemispheres==-1) & (powers==p)},freeP_unilateral);
    [paramsR,dParamsR] = get_delta_fitted(extracted.data{(subjects==subject) & (hemispheres==1) & (powers==p)},freeP_unilateral);
    % sum the delta bias for each uni 
    dParams = (dParamsR+dParamsL).*freeP_unilateral; 

    dLs(s) = dParamsL(1); 
    dRs(s) = dParamsR(1); 
    dParamsum(s) = dParams(1);
    % fit control for bi
    biBlock = extracted.data{(subjects==subject) & (hemispheres==0) & (powers==p)};
    controlBlock = filterStructRows(biBlock, ~biBlock.is_laserTrial);
    optoBlock = filterStructRows(biBlock, biBlock.is_laserTrial); 
    controlfit = plts.behaviour.GLMmulti(controlBlock, 'simpLogSplitVSplitA');
    controlfit.fit;
    
%     if should_plot
%         figure; 
%         plotParams.LineStyle = '-';
%         plotParams.DotStyle = ['.'];
%         plotParams.MarkerSize = 24; 
%         plot_optofit(controlfit,plotParams,plotfit)
%         hold on; 
%     end 

    % fit opto with control+delta bias 
    optoBlock.freeP = logical([0,0,0,0,0,0]);
    orifit = plts.behaviour.GLMmulti(optoBlock, 'simpLogSplitVSplitA');
    
    orifit.prmInit = controlfit.prmFits+dParams;
    orifit.fitCV(5); 
    lldelta(s)=mean(orifit.logLik);


    if should_plot
       f=figure; 
       f.Position = [10,10,300,300];
       plotParams.plottype = 'log'; 
       plotParams.LineStyle = ':';
       plotParams.DotStyle = '.';

       plotParams.MarkerSize = 36; 
       plot_optofit(orifit,plotParams,plotfit,controlfit.prmFits(4)); 
       hold on; 

       title(sprintf('%s, %.0f mW', ...
             subject,p))
    end 

    % refit opto with full
    optoBlock.freeP = logical([1,1,1,0,1,1]);
    %optoBlock.freeP = logical([1,0,0,0,0,0]);
    orifit = plts.behaviour.GLMmulti(optoBlock, 'simpLogSplitVSplitA');
    orifit.prmInit = controlfit.prmFits;
    orifit.fitCV(5); 
    deltaParams = mean(orifit.prmFits,1);
    dParamsBoth(s)=deltaParams(1);
%     if should_plot
%        plotParams.LineStyle = '--';
%        plotParams.DotStyle = 'none';
%        plotParams.MarkerSize = 8; 
%        plot_optofit(orifit,plotParams,plotfit,controlfit.prmFits(4))
%     end 

    llfull(s)=mean(orifit.logLik);




end

%%
figure; plot(lldelta(1:2),llfull(1:2),'.',MarkerSize=30); hold on; 
plot(lldelta(3:4),llfull(3:4),'.',MarkerSize=30); 
hold on; plot([0,1],[0,1],'k--');
xlabel('control+deltaBias')
ylabel('full')
title('-veLogLikelihood')

%%
figure;
plot(dParamsum,dParamsBoth,'.',MarkerSize=30);
hold on; plot([-3,3],[-3,3],'k--');

hold on; plot(dLs,[0,0,0],'.');

hold on; plot(dRs,[0,0,0],'.');



%% 
function [pR] = get_performance(ev)
% varargin for the visDiff and the audDiff such that we get all the inputs

visDiff = int8(ev.stim_visDiff*100);
audDiff = ev.stim_audDiff;
% hardcode, otherwise he combinations will depend on ev and the output
% matrix will be of varaiable size
visStim = [-40,-20,-10,0,10,20,40];
audStim = [-60,0,60]; 
[visGrid, audGrid] = meshgrid(visStim,audStim);%
response_per_condition = arrayfun(@(x,y) ev.response_direction(ismember([visDiff,audDiff],[x,y],'rows')), visGrid, audGrid,'UniformOutput',0);
pR = cellfun(@(x) mean(x-1),response_per_condition); 
pR  = log(pR./(1-pR));
minN =2;
n_per_cond = cellfun(@(x) numel(x),response_per_condition); 
pR(n_per_cond<minN) = nan;
end 

function [delta] = get_delta_choice(ev)
optoBlock = filterStructRows(ev, ev.is_laserTrial); 
controlBlock = filterStructRows(ev, ~ev.is_laserTrial);
delta=get_performance(optoBlock)-get_performance(controlBlock);
end 

function plot_additive_pred(biBlock,deltas)
optoBlock = filterStructRows(biBlock, biBlock.is_laserTrial); 
controlBlock = filterStructRows(biBlock, ~biBlock.is_laserTrial);

plotOpt.MarkerSize = 12; 
plotOpt.FaceAlpha = 0.1; 

pOpto = get_performance(optoBlock);
pCtrl = get_performance(controlBlock);
pPred = pCtrl+deltas; 

% plot

visValues = [-40,-20,-10,0,10,20,40];

figure;
lineColors = plts.general.selectRedBlueColors([-1,0,1]);

plotOpt.lineStyle = '-';
plotOpt.Marker = '.';
plts.general.rowsOfGrid(visValues, pOpto, lineColors, plotOpt);
plotOpt.lineStyle = '--';
plotOpt.Marker = '*';
plts.general.rowsOfGrid(visValues, pPred, lineColors, plotOpt);
%plts.general.rowsOfGrid(visValues, pCtrl, lineColors, plotOpt);

end 

function [ctrlParams,deltaParams] = get_delta_fitted(currBlock,freePs)
    optoBlock = filterStructRows(currBlock, currBlock.is_laserTrial); 
    controlBlock = filterStructRows(currBlock, ~currBlock.is_laserTrial);

    controlfit = plts.behaviour.GLMmulti(controlBlock, 'simpLogSplitVSplitA');
    controlfit.fit;

    % optofit with bias delta
    optoBlock.freeP = freePs;
    orifit = plts.behaviour.GLMmulti(optoBlock, 'simpLogSplitVSplitA');
    orifit.prmInit = controlfit.prmFits;
    orifit.fitCV(5); 
    ctrlParams = controlfit.prmFits; 
    deltaParams = mean(orifit.prmFits,1);
end 

% plotting function 
