%

% varargin = ['sepPlots', {nan}, varargin];
% varargin = ['expDef', {'t'}, varargin];
% varargin = ['plotType', {'res'}, varargin];
% varargin = ['noPlot', {0}, varargin];
params.subject  = {['AV030']};
params.expDate  = 'last15';
params.checkEvents = '1';

plts.behaviour.glmFit(params,sepPLots='1');
extracted = plts.behaviour.getTrainingData(params);
%%
ev = concatenateEvents(extracted.data);
% plot 

figure;
plotOpt.lineStyle = '-'; %plotParams.LineStyle;
plotOpt.Marker = '*';
chrono_correct = get_rts(filterStructRows(ev,(ev.response_feedback==1 & ...
    ev.is_validTrial & ((ev.stim_laser1_power+ev.stim_laser2_power)==0))),plotOpt); 

%% 
ev