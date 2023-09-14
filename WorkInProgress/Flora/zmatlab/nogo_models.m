clc; clear all;
extracted = loadOptoData('balanceTrials',0,'sepMice',1,'reExtract',1,'sepHemispheres',1); 
%%

%s = 11; 

for s=1:numel(extracted.subject)
currBlock = extracted.data{s};
nTrials(s) = numel(currBlock.is_blankTrial); 
optoBlock = filterStructRows(currBlock, currBlock.is_laserTrial); 
controlBlock = filterStructRows(currBlock, ~currBlock.is_laserTrial);


%
figure;
plotOpt.lineStyle = '-'; %plotParams.LineStyle;
plotOpt.Marker = 'none';
plotOpt.toPlot = 1;

% well technically all we need is the performance per condition
[pR,pNG] = get_nogo_performance(controlBlock,plotOpt);
pL= 1-pR-pNG; 
%
hold on; 
plotOpt.lineStyle = '--'; %plotParams.LineStyle;
plotOpt.Marker = '.';
plotOpt.MarkerSize = 30; 

[pR_,pNG_] = get_nogo_performance(optoBlock,plotOpt); 
ylim([0,.25])

%
% figure; 
% visStim = [-40,-20,-10,0,10,20,40];
% audStim = [-60,0,60]; 
% [visGrid, audGrid] = meshgrid(visStim,audStim);%
% lineColors = plts.general.selectRedBlueColors(audGrid(:,1));
% plts.general.rowsOfGrid(visStim, pNG_-pNG, lineColors,plotOpt);
% 
title(sprintf('%s,hemisphere:%.0d',extracted.subject{s},extracted.hemisphere{s}))


diff_(s,:,:) = (pNG_-pNG); 
end
%%
figure; 
plts.general.rowsOfGrid(visStim, permute(diff_,[2,3,1]), lineColors,plotOpt);
yline(0)
%%

function [pR,pNG] = get_nogo_performance(ev,plotOpt)
% varargin for the visDiff and the audDiff such that we get all the inputs

visDiff = int8(ev.stim_visDiff*100);
audDiff = ev.stim_audDiff;
% hardcode, otherwise he combinations will depend on ev and the output
% matrix will be of varaiable size
visStim = [-40,-20,-10,0,10,20,40];
audStim = [-60,0,60]; 
[visGrid, audGrid] = meshgrid(visStim,audStim);%
response_per_condition = arrayfun(@(x,y) ev.response_direction(ismember([visDiff,audDiff],[x,y],'rows')), visGrid, audGrid,'UniformOutput',0);
pR = cellfun(@(x) numel(x(x==2))/numel(x),response_per_condition); 
pNG = cellfun(@(x) numel(x(x==0))/numel(x),response_per_condition); 

%pR  = log(pR./(1-pR));
minN =2;
n_per_cond = cellfun(@(x) numel(x),response_per_condition); 
pR(n_per_cond<minN) = nan;
pNG(n_per_cond<minN) = nan;




if plotOpt.toPlot
lineColors = plts.general.selectRedBlueColors(audGrid(:,1));
plts.general.rowsOfGrid(visStim, pNG, lineColors,plotOpt);
end 

end 

