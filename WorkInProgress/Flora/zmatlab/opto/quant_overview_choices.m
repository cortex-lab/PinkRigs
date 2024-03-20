%% 

clc; clear all;
extracted = loadOptoData('balanceTrials',0,'sepMice',1,'reExtract',1,'sepHemispheres',1,'whichSet', 'bi_high');
%%

% fig -- choice change per trial types

trialname = {'aud';'vis';'coherent';'conflict'};

plotOpt.toPlot=0; 

for w=1:numel(trialname)
    for s=1:numel(extracted.subject)    
        ev = extracted.data{s};
        which = {ev.is_auditoryTrial;ev.is_visualTrial;ev.is_coherentTrial;ev.is_conflictTrial}; 

        pRo(w,s) =nanmean(get_metrics_StimClass(filterStructRows(ev,(which{w} & ev.is_laserTrial)),'pR',plotOpt),'all');
        pR(w,s) =nanmean(get_metrics_StimClass(filterStructRows(ev,(which{w} & ~ev.is_laserTrial)),'pR',plotOpt),'all');
    
        dpR(w,s) = nanmean(get_metrics_StimClass(filterStructRows(ev,(which{w} & ev.is_laserTrial)),'pR',plotOpt) - ...
                             get_metrics_StimClass(filterStructRows(ev,(which{w} & ~ev.is_laserTrial)),'pR',plotOpt),'all');
    

        postlaser(s) = nanmean(get_metrics_StimClass(filterStructRows(ev,(ev.postLaserTrial & ~ev.is_laserTrial)),'pR',plotOpt) - ...
                             get_metrics_StimClass(filterStructRows(ev,(~ev.postLaserTrial & ~ev.is_laserTrial)),'pR',plotOpt),'all');
    end 
end


%%

figure; 
allplot = [dpR;postlaser]; 
h = plot(allplot,'Color',[0.5, 0.5, 0.5]);
hold on; 
%plot(mean(allplot,2),'k','LineWidth',6); 
xticks(1:5)
ylim([-0.7,0.7])
trialname{5} = 'n+1';
xticklabels(trialname)
hline(0,'k--')
%%