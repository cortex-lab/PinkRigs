function [median_rt] = get_rts(ev,rtType,plotOpt)
% varargin for the visDiff and the audDiff such that we get all the inputs 
%ev.rt = ev.timeline_choiceMoveOn-min([ev.timeline_audPeriodOn,ev.timeline_visPeriodOn],[],2);

if strcmp('rtAud',rtType)
    ev.rt = ev.timeline_choiceMoveOn-ev.timeline_audPeriodOn;
elseif strcmp('rtMin',rtType)
    ev.rt = ev.timeline_choiceMoveOn-min([ev.timeline_audPeriodOn,ev.timeline_visPeriodOn],[],2);
elseif strcmp('rtThresh',rtType) % reaction time calculated when decision time is reache
    ev.rt = ev.timeline_choiceThreshOn - ev.timeline_audPeriodOn; 
end 

visDiff = int8(ev.stim_visDiff*100);
audDiff = ev.stim_audDiff;

rt = ev.rt;

% hardcode, otherwise he combinations will depend on ev and the output
% matrix will be of varaiable size

visStim = [-40,-20,-10,0,10,20,40];
audStim = [-60,0,60]; 

[visGrid, audGrid] = meshgrid(visStim,audStim);
%
rt_per_cond = arrayfun(@(x,y) rt(ismember([visDiff,audDiff],[x,y],'rows') & ~isnan(rt)), visGrid, audGrid,'UniformOutput',0);
median_rt = cellfun(@(x) median(x),rt_per_cond); 
mad_rt = cellfun(@(x) mad(x),rt_per_cond); 
minN =5;
n_per_cond = cellfun(@(x) numel(x),rt_per_cond); 
median_rt(n_per_cond<minN) = nan;

% add an option to plot the chronometric curve

% plotOpt.lineStyle = 'none'; %plotParams.LineStyle;
% plotOpt.Marker = '.';
% plotOpt.MarkerSize = 24;

if plotOpt.toPlot
lineColors = plts.general.selectRedBlueColors(audGrid(:,1));
plts.general.rowsOfGrid(visStim, median_rt, lineColors,plotOpt);
end 


end 