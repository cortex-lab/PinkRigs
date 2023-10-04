
clc; clear all;
extracted = loadOptoData('balanceTrials',0,'sepMice',1,'reExtract',0,'sepHemispheres',1); 



%%
s=1;
plotOpt.toPlot=0; 
 
ev = extracted.data{s};
ev.rt = ev.timeline_choiceMoveOn-min([ev.timeline_audPeriodOn,ev.timeline_visPeriodOn],[],2);
%ev.rt = ev.timeline_choiceMoveOn-ev.timeline_audPeriodOn; 

% get contra and ipsi trials for laser and non-laser conditions
ev.laser_stimDiff = min([ev.timeline_audPeriodOn,ev.timeline_visPeriodOn],[],2)-ev.timeline_laserOn_rampStart;
    
opto = get_rts(filterStructRows(ev,((ev.response_feedback==1) & ev.is_laserTrial)),'rtThresh',plotOpt); 

ctrl = get_rts(filterStructRows(ev,((ev.response_feedback==1) & ~ev.is_laserTrial)),'rtThresh',plotOpt); 

%

figure;
plot(ctrl(2,:)); 
hold on; 
plot(opto(2,:)); 
