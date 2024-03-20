function [avg_metric] = get_metrics_StimClass(ev,metricType,plotOpt)
% varargin for the visDiff and the audDiff such that we get all the inputs 
%ev.rt = ev.timeline_choiceMoveOn-min([ev.timeline_audPeriodOn,ev.timeline_visPeriodOn],[],2);


% the stimulus grid
% hardcode, otherwise he combinations will depend on ev and the output
% matrix will be of varaiable size
visStim = [-40,-20,-10,0,10,20,40];
audStim = [-60,0,60]; 
[visGrid, audGrid] = meshgrid(visStim,audStim);
minN =5;  % the minimum number of trials in each class
visDiff = int8(ev.stim_visDiff*100);
audDiff = ev.stim_audDiff;


if strcmp('pR',metricType)
    metric = ev.response_direction==2; 
elseif strcmp('pNoGo',metricType)
    metric = ev.response_direction==0;
elseif contains(metricType,'rt')   
    if strcmp('rtAud',metricType)
        ev.rt = ev.timeline_choiceMoveOn-ev.timeline_audPeriodOn;
    elseif strcmp('rtMin',metricType)
        ev.rt = ev.timeline_choiceMoveOn-min([ev.timeline_audPeriodOn,ev.timeline_visPeriodOn],[],2);
    elseif strcmp('rtThresh',metricType) % reaction time calculated when decision time is reache
        ev.rt = ev.timeline_choiceThreshOn - ev.timeline_audPeriodOn; 
    end 
    metric = ev.rt;    
end 

% calculate this per cpndition
metric_per_cond = arrayfun(@(x,y) metric(ismember([visDiff,audDiff],[x,y],'rows') & ~isnan(metric)), visGrid, audGrid,'UniformOutput',0);

% now we take the average per cond  -- for nominal data we take the mean,
% for rt, we take the median 
if islogical(metric)
    avg_metric = cellfun(@(x) mean(x),metric_per_cond); 
else
    avg_metric = cellfun(@(x) median(x),metric_per_cond);
end 

n_per_cond = cellfun(@(x) numel(x),metric_per_cond); 
avg_metric(n_per_cond<minN) = nan;

% add an option to plot the stimOption dependent curve curve
if plotOpt.toPlot
lineColors = plts.general.selectRedBlueColors(audGrid(:,1));
plts.general.rowsOfGrid(visStim, avg_metric, lineColors,plotOpt);
end 


end 