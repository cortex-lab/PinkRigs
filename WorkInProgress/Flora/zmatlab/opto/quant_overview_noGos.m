clc; clear all;
extracted = loadOptoData('balanceTrials',0,'sepMice',1,'reExtract',1,'sepHemispheres',1,'whichSet', 'bi_high_nogo');





plotOpt.toPlot=0; 

for s=1:numel(extracted.subject)    
    ev = extracted.data{s};
    which = {ev.is_auditoryTrial;ev.is_visualTrial;ev.is_coherentTrial;ev.is_conflictTrial}; 



    dpR(s) = nanmean(get_metrics_StimClass(filterStructRows(ev,(ev.is_laserTrial)),'pNoGo',plotOpt) - ...
                         get_metrics_StimClass(filterStructRows(ev,(~ev.is_laserTrial)),'pNoGo',plotOpt),'all');

    postlaser(s) = nanmean(get_metrics_StimClass(filterStructRows(ev,(ev.postLaserTrial & ~ev.is_laserTrial)),'pNoGo',plotOpt) - ...
                         get_metrics_StimClass(filterStructRows(ev,(~ev.postLaserTrial & ~ev.is_laserTrial)),'pNoGo',plotOpt),'all');
end 
%%

figure; 
allplot = [dpR;postlaser]; 
h = plot(allplot,'Color',[0.5, 0.5, 0.5]);
hold on; 
%plot(mean(allplot,2),'k','LineWidth',6); 
xticks(1:2)
ylim([-0.1,0.3])
xticklabels({'laser','n+1'})
hline(0,'k--')