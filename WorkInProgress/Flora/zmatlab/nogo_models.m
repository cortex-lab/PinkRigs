clc; clear all;
extracted = loadOptoData('balanceTrials',0,'sepMice',1,'reExtract',0,'sepHemispheres',1); 
%%

s = 1 ; 
currBlock = extracted.data{s};
nTrials(s) = numel(currBlock.is_blankTrial); 
optoBlock = filterStructRows(currBlock, currBlock.is_laserTrial); 
controlBlock = filterStructRows(currBlock, ~currBlock.is_laserTrial);

% well technically all we need is the performance per condition
[pR,pNG] = get_nogo_performance(controlBlock);
pL= 1-pR-pNG; 
%% 

[pR_,pNG_] = get_nogo_performance(optoBlock); 

%% 

figure
%-- Plot the axis system
[h,hg,htick]=terplot;
%-- Plot the data ...
hter1=ternaryc(pR(1:3,:)',pL(1:3,:)',pNG(1:3,:)');

%hter2=ternaryc(pR(2,:),pL(2,:),pNG(2,:));

%hter3=ternaryc(pR(3,:),pL(3,:),pNG(3,:));

% %-- ... and modify the symbol:
% set(hter1,'marker','o','markerfacecolor','none','markersize',4)
% set(hter2,'marker','x','markerfacecolor','none','markersize',4)

hlabels=terlabel('pR','pL','pNG');
%%

function [pR,pNG] = get_nogo_performance(ev)
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

end 

