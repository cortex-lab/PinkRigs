clc; clear all;
extracted = loadOptoData('balanceTrials',0,'sepMice',1,'reExtract',1,'sepHemispheres',1,'whichSet', 'uni_all_nogo');



%%
which_set='rtThresh';
plotOpt.toPlot=0; 
for s=1:numel(extracted.subject)    
    ev = extracted.data{s};
    which = {ev.is_auditoryTrial;ev.is_visualTrial;ev.is_coherentTrial;ev.is_conflictTrial}; 



    dpR(s) = nanmean(get_metrics_StimClass(filterStructRows(ev,(ev.response_direction==2 & ev.is_laserTrial)),which_set,plotOpt) - ...
                         get_metrics_StimClass(filterStructRows(ev,(ev.response_direction==2 & ~ev.is_laserTrial)),which_set,plotOpt),'all');

    dpL(s) = nanmean(get_metrics_StimClass(filterStructRows(ev,(ev.response_direction==1 & ev.is_laserTrial)),which_set,plotOpt) - ...
                         get_metrics_StimClass(filterStructRows(ev,(ev.response_direction==1 & ~ev.is_laserTrial)),which_set,plotOpt),'all');

    postlaserR(s) = nanmean(get_metrics_StimClass(filterStructRows(ev,(ev.response_direction==2 & ev.postLaserTrial & ~ev.is_laserTrial)),which_set,plotOpt) - ...
                         get_metrics_StimClass(filterStructRows(ev,(ev.response_direction==2 & ~ev.postLaserTrial & ~ev.is_laserTrial)),which_set,plotOpt),'all');

    postlaserL(s) = nanmean(get_metrics_StimClass(filterStructRows(ev,(ev.response_direction==1 & ev.postLaserTrial & ~ev.is_laserTrial)),which_set,plotOpt) - ...
                     get_metrics_StimClass(filterStructRows(ev,(ev.response_direction==1 & ~ev.postLaserTrial & ~ev.is_laserTrial)),which_set,plotOpt),'all');
end 
%%


figure;

subplot(1,2,1)
% postlaserR;postlaserL
allplot = [ dpR;dpL]; 
h = plot(allplot,'Color',[0.5, 0.5, 0.5]);
hold on; 
%plot(mean(allplot,2),'k','LineWidth',6); 
xticks(1:2)
ylim([-0.2,0.2])
xticklabels({'ipsi','contra'})
hline(0,'k--')

subplot(1,2,2)
% postlaserR;postlaserL
allplot = [ postlaserR;postlaserL]; 
h = plot(allplot,'Color',[0.5, 0.5, 0.5]);
hold on; 
%plot(mean(allplot,2),'k','LineWidth',6); 
xticks(1:2)
ylim([-0.2,0.2])
xticklabels({'ipsi,n+1','contra,n+1'})
hline(0,'k--')


