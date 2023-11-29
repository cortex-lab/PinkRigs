%

% varargin = ['sepPlots', {nan}, varargin];
% varargin = ['expDef', {'t'}, varargin];
% varargin = ['plotType', {'res'}, varargin];
% varargin = ['noPlot', {0}, varargin];
params.subject  = {['AV034']};
params.expDate  = 'last25';
params.checkEvents = '1';

plts.behaviour.glmFit(params);
extracted = plts.behaviour.getTrainingData(params);
%%
ev = concatenateEvents(extracted.data);
% plot 

figure;
plotOpt.lineStyle = '-'; %plotParams.LineStyle;
plotOpt.Marker = '*';
plotOpt.toPlot = 1;

chrono_correct = get_rts(filterStructRows(ev,(ev.response_feedback==1 & ...
    ev.is_validTrial & ((ev.stim_laser1_power+ev.stim_laser2_power)==0))),'rtThresh',plotOpt); 

ylabel('median RT')
xlabel('contrast')
title(sprintf('%s',params.subject{1}))
%% 
