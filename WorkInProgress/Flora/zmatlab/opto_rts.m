% script to calculate reaction times for each 
% mouse/power etc

clc; clear all;
extracted = loadOptoData('balanceTrials',0,'sepMice',1,'reExtract',1,'sepHemispheres',1); 

%%
for s=1:numel(extracted.subject)    
    ev = extracted.data{s};
    ev.rt = ev.timeline_choiceMoveOn-min([ev.timeline_audPeriodOn,ev.timeline_visPeriodOn]')';
    % get contra and ipsi trials for laser and non-laser conditions
    ev.laser_stimDiff = min([ev.timeline_audPeriodOn,ev.timeline_visPeriodOn]')'-ev.timeline_laserOn_rampStart;


        
    c = get_rts(filterStructRows(ev,(ev.timeline_choiceMoveDir==1 & ev.is_laserTrial))) - ...
        get_rts(filterStructRows(ev,(ev.timeline_choiceMoveDir==1 & ~ev.is_laserTrial))); 
    
    i = get_rts(filterStructRows(ev,(ev.timeline_choiceMoveDir==2 & ev.is_laserTrial))) - ...
        get_rts(filterStructRows(ev,(ev.timeline_choiceMoveDir==2 & ~ev.is_laserTrial))); 

    slow_ = get_rts(filterStructRows(ev,(ev.timeline_choiceMoveDir==2 & ev.is_laserTrial & ...
        ev.laser_stimDiff>nanmedian(ev.laser_stimDiff)))); 

    fast_ = get_rts(filterStructRows(ev,(ev.timeline_choiceMoveDir==2 & ev.is_laserTrial & ...
    ev.laser_stimDiff<nanmedian(ev.laser_stimDiff)))); 

    
    ipsi(s) = nanmean(i,'all');
    contra(s) =  nanmean(c,'all');

    slow(s) = nanmean(slow_,'all');
    fast(s) = nanmean(fast_,'all');

end
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

function [median_rt] = get_rts(ev)
% varargin for the visDiff and the audDiff such that we get all the inputs 

visDiff = int8(ev.stim_visDiff*100);
audDiff = ev.stim_audDiff;

rt = ev.rt;

% hardcode, otherwise he combinations will depend on ev and the output
% matrix will be of varaiable size

visStim = [-40,-20,-10,0,10,20,40];
audStim = [-60,0,60]; 

[visGrid, audGrid] = meshgrid(visStim,audStim);
%
rt_per_cond = arrayfun(@(x,y) rt(ismember([visDiff,audDiff],[x,y],'rows') & ~isnan(rt)), visGrid, audGrid,'UniformOutput',0);
median_rt = cellfun(@(x) median(x),rt_per_cond); 
minN =10;
n_per_cond = cellfun(@(x) numel(x),rt_per_cond); 
median_rt(n_per_cond<minN) = nan;
end 