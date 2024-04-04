clc; clear all;
extracted = loadOptoData('balanceTrials',0,'sepMice',1,'reExtract',1,'sepHemispheres',1,'whichSet', 'uni_all_nogo');




%%
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


%% unilateral version when I might want to 




which_set='pNoGo';
plotOpt.toPlot=0; 
for s=1:numel(extracted.subject)    
    ev = extracted.data{s};
    ev.stim_direction = sign(sign(ev.stim_visDiff)+sign(ev.stim_audDiff));

    which = {ev.is_auditoryTrial;ev.is_visualTrial;ev.is_coherentTrial;ev.is_conflictTrial}; 



    dpR(s) = nanmean(get_metrics_StimClass(filterStructRows(ev,(ev.stim_direction==1 & ev.is_laserTrial & ev.is_auditoryTrial)),which_set,plotOpt) - ...
                         get_metrics_StimClass(filterStructRows(ev,(ev.stim_direction==1 & ~ev.is_laserTrial & ev.is_auditoryTrial)),which_set,plotOpt),'all');

    dpL(s) = nanmean(get_metrics_StimClass(filterStructRows(ev,(ev.stim_direction==-1 & ev.is_laserTrial & ev.is_auditoryTrial)),which_set,plotOpt) - ...
                         get_metrics_StimClass(filterStructRows(ev,(ev.stim_direction==-1 & ~ev.is_laserTrial & ev.is_auditoryTrial)),which_set,plotOpt),'all');


    dpBlank(s) = nanmean(get_metrics_StimClass(filterStructRows(ev,(ev.is_blankTrial & ev.is_laserTrial)),which_set,plotOpt) - ...
                         get_metrics_StimClass(filterStructRows(ev,(ev.is_blankTrial & ~ev.is_laserTrial)),which_set,plotOpt),'all');


    postlaserR(s) = nanmean(get_metrics_StimClass(filterStructRows(ev,(ev.stim_direction==1 & ev.postLaserTrial & ~ev.is_laserTrial & ev.is_auditoryTrial)),which_set,plotOpt) - ...
                         get_metrics_StimClass(filterStructRows(ev,(ev.stim_direction==1 & ~ev.postLaserTrial & ~ev.is_laserTrial & ev.is_auditoryTrial)),which_set,plotOpt),'all');

    postlaserL(s) = nanmean(get_metrics_StimClass(filterStructRows(ev,(ev.stim_direction==-1 & ev.postLaserTrial & ~ev.is_laserTrial & ev.is_auditoryTrial)),which_set,plotOpt) - ...
                     get_metrics_StimClass(filterStructRows(ev,(ev.stim_direction==-1 & ~ev.postLaserTrial & ~ev.is_laserTrial & ev.is_auditoryTrial)),which_set,plotOpt),'all');

    postlaserBlank(s) = nanmean(get_metrics_StimClass(filterStructRows(ev,(ev.is_blankTrial & ev.postLaserTrial & ~ev.is_laserTrial)),which_set,plotOpt) - ...
                 get_metrics_StimClass(filterStructRows(ev,(ev.is_blankTrial & ~ev.postLaserTrial & ~ev.is_laserTrial)),which_set,plotOpt),'all');


end 
%%

figure;

subplot(1,2,1)
% postlaserR;postlaserL
allplot = [dpBlank;dpR;dpL]; 
h = plot(allplot,'Color',[0.5, 0.5, 0.5]);
hold on; 
%plot(mean(allplot,2),'k','LineWidth',6); 
xticks(1:3)
ylim([-0.25,0.4])
xticklabels({'blank','ipsi','contra'})
hline(0,'k--')

subplot(1,2,2)
% postlaserR;postlaserL
allplot = [ postlaserBlank;postlaserR;postlaserL]; 
h = plot(allplot,'Color',[0.5, 0.5, 0.5]);
hold on; 
%plot(mean(allplot,2),'k','LineWidth',6); 
xticks(1:3)
ylim([-0.25,0.4])
xticklabels({'blank,n+1','ipsi,n+1','contra,n+1'})
hline(0,'k--')
%% OR



figure;

subplot(1,3,1)
% postlaserR;postlaserL
allplot = [dpBlank;postlaserBlank]; 
h = plot(allplot,'Color',[0.5, 0.5, 0.5]);
hold on; 
%plot(mean(allplot,2),'k','LineWidth',6); 
xticks(1:2)
ylim([-0.25,0.4])
xticklabels({'blank','blank,n+1'})
hline(0,'k--')

subplot(1,3,2)
% postlaserR;postlaserL
allplot = [dpR;postlaserR]; 
h = plot(allplot,'Color',[0.5, 0.5, 0.5]);
hold on; 
%plot(mean(allplot,2),'k','LineWidth',6); 
xticks(1:2)
ylim([-0.25,0.4])
xticklabels({'ipsi','ipsi,n+1'})
hline(0,'k--')

subplot(1,3,3)
% postlaserR;postlaserL
allplot = [dpL;postlaserL]; 
h = plot(allplot,'Color',[0.5, 0.5, 0.5]);
hold on; 
%plot(mean(allplot,2),'k','LineWidth',6); 
xticks(1:2)
ylim([-0.25,0.4])
xticklabels({'contra','contra,n+1'})
hline(0,'k--')

