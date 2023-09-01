% script to calculate reaction times for each 
% mouse/power etc

clc; clear all;
extracted = loadOptoData('balanceTrials',0,'sepMice',1,'reExtract',1,'sepHemispheres',1); 

%%
plotOpt.toPlot=0; 
for s=1:numel(extracted.subject)    
    ev = extracted.data{s};
    ev.rt = ev.timeline_choiceMoveOn-min([ev.timeline_audPeriodOn,ev.timeline_visPeriodOn],[],2);
    %ev.rt = ev.timeline_choiceMoveOn-ev.timeline_audPeriodOn; 

    % get contra and ipsi trials for laser and non-laser conditions
    ev.laser_stimDiff = min([ev.timeline_audPeriodOn,ev.timeline_visPeriodOn],[],2)-ev.timeline_laserOn_rampStart;
        
    c = get_rts(filterStructRows(ev,(ev.timeline_choiceMoveDir==1 & ev.is_laserTrial)),plotOpt) - ...
        get_rts(filterStructRows(ev,(ev.timeline_choiceMoveDir==1 & ~ev.is_laserTrial)),plotOpt); 
    
    i = get_rts(filterStructRows(ev,(ev.timeline_choiceMoveDir==2 & ev.is_laserTrial)),plotOpt) - ...
        get_rts(filterStructRows(ev,(ev.timeline_choiceMoveDir==2 & ~ev.is_laserTrial)),plotOpt); 

    slow_ = get_rts(filterStructRows(ev,(ev.timeline_choiceMoveDir==2 & ev.is_laserTrial & ...
        ev.laser_stimDiff>nanmedian(ev.laser_stimDiff))),plotOpt); 

    fast_ = get_rts(filterStructRows(ev,(ev.timeline_choiceMoveDir==2 & ev.is_laserTrial & ...
    ev.laser_stimDiff<nanmedian(ev.laser_stimDiff))),plotOpt); 

    
    ipsi(s) = nanmean(i,'all');
    contra(s) =  nanmean(c,'all');
    slow(s) = nanmean(slow_,'all');
    fast(s) = nanmean(fast_,'all');

end
%%
figure; 
plot([1,2],[ipsi;contra]); hold on;
plot([1,2],[0,0],'k--')
%% plot the actual chronometric curves
%
s=3; 
plotOpt.toPlot=1; 

ev = extracted.data{s};
%ev.rt = ev.timeline_choiceMoveOn-min([ev.timeline_audPeriodOn,ev.timeline_visPeriodOn]')';
ev.rt = ev.timeline_choiceMoveOn-ev.timeline_audPeriodOn; 
% get contra and ipsi trials for laser and non-laser conditions
ev.laser_stimDiff = min([ev.timeline_audPeriodOn,ev.timeline_visPeriodOn]')'-ev.timeline_laserOn_rampStart;
        

figure; 
%plot(chrono_correct')
plotOpt.lineStyle = '--'; %plotParams.LineStyle;
plotOpt.Marker = '+';
plotOpt.MarkerSize = 12;

chrono_correct = get_rts(filterStructRows(ev,(ev.response_feedback==1 & ev.is_laserTrial)),plotOpt); 

plotOpt.lineStyle = '-'; %plotParams.LineStyle;
plotOpt.Marker = '*';
chrono_correct = get_rts(filterStructRows(ev,(ev.response_feedback==1 & ~ev.is_laserTrial)),plotOpt); 


%%
figure;
is_high = cellfun(@(x) (x==17),extracted.power); 
is_low = cellfun(@(x) (x==10),extracted.power); 
plot([1,2],[ipsi(is_low);contra(is_low)],color='b'); 
hold on 
plot([1,2],[ipsi(is_high);contra(is_high)],color='r'); 
[~,p_ic]= ttest(ipsi,contra);

%%
figure;
plot([1,2],[slow;fast],color='b'); 
[~,p_sf]= ttest(slow,fast);


%%
figure; plot([1,2],[contra;ipsi]);
ylabel('rel RT, opto (s)')
xticks([1,2])
xticklabels({'left','right'})
xlabel('choice dir')
%%

