clc; clear all;
extracted = loadOptoData('balanceTrials',0,'sepMice',1,'reExtract',1,... 
                'sepHemispheres',0,'sepPowers',0,'sepDiffPowers',0,'whichSet', 'bi_single_example');
%%
    
events = extracted.data{1};


sessions = unique(events.sessionID); 
plotOpt.toPlot=0; 

for s=1:numel(sessions)
    ev = filterStructRows(events,(events.sessionID==sessions(s)));     
    dpR(s) = nanmean(get_metrics_StimClass(filterStructRows(ev,(ev.is_laserTrial)),'pR',plotOpt) - ...
                         get_metrics_StimClass(filterStructRows(ev,(~ev.is_laserTrial)),'pR',plotOpt),'all');

    lasertrials = filterStructRows(ev,(ev.is_laserTrial)); 
    hemi(s) = mean(lasertrials.stim_laserPosition_);
end
%% 
f=figure; 
f.Position = [10,10,900,300];

plot(dpR,'-','Color',[.5,.5,.5]);
hold on; 


unique_hemis = unique(hemi); 

csets={[0, 102, 204];[0, 153, 0];[255, 128, 0]}; 

for i=1:numel(unique_hemis)
    my_hemi=unique_hemis(i);
    colored = nan(1,numel(dpR));
    colored(hemi==my_hemi) = dpR(hemi==my_hemi);
    plot(colored,'.','Color',csets{i}/255,'MarkerSize',25);
    hold on;
end 
hline(0,'k--')
xlabel('sessionID')
xlim([-1,85])
ylim([-.6,.6])