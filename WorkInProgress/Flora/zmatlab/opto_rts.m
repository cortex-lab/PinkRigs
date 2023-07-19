% script to calculate reaction times for each 
% mouse/power etc
clc; clear all;
extracted = loadOptoData('balanceTrials',0,'sepMice',1,'reExtract',0,'sepHemispheres',0); 

%%
for s=1:numel(extracted.subject)    
    ev = extracted.data{s};
    ev.rt = ev.timeline_choiceMoveOn-min([ev.timeline_audPeriodOn,ev.timeline_visPeriodOn]')';
    % get contra and ipsi trials for laser and non-laser conditions

    c = get_rts(filterStructRows(ev,(ev.timeline_choiceMoveDir==1 & ev.is_laserTrial))) - ...
        get_rts(filterStructRows(ev,(ev.timeline_choiceMoveDir==1 & ~ev.is_laserTrial))); 
    
    i = get_rts(filterStructRows(ev,(ev.timeline_choiceMoveDir==2 & ev.is_laserTrial))) - ...
        get_rts(filterStructRows(ev,(ev.timeline_choiceMoveDir==2 & ~ev.is_laserTrial))); 

    ipsi(s) = nanmean(i,'all');
    contra(s) =  nanmean(c,'all');


end
%%
figure;
is_high = cellfun(@(x) (x==17),extracted.power); 
is_low = cellfun(@(x) (x==10),extracted.power); 
plot([1,2],[ipsi(is_low);contra(is_low)],color='b'); 
hold on 
plot([1,2],[ipsi(is_high(1:5));contra(is_high(1:5))],color='r'); 




%%

function [median_rt] = get_rts(ev)
visDiff = ev.stim_visDiff;
audDiff = ev.stim_audDiff;

rt = ev.rt;
[visGrid, audGrid] = meshgrid(unique(visDiff),unique(audDiff));
%
rt_per_cond = arrayfun(@(x,y) rt(ismember([visDiff,audDiff],[x,y],'rows') & ~isnan(rt)), visGrid, audGrid,'UniformOutput',0);
median_rt = cellfun(@(x) median(x),rt_per_cond); 
minN = 10;
n_per_cond = cellfun(@(x) numel(x),rt_per_cond); 
median_rt(n_per_cond<minN) = nan;
end 