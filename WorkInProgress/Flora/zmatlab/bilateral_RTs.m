%%
clc; clear all;
extracted = loadOptoData('balanceTrials',0,'sepMice',1,'reExtract',1,'sepHemispheres',1); 

% prepare params so that they can be matched
powers = [extracted.power{:}]; 
powers([extracted.hemisphere{:}]==0) = powers([extracted.hemisphere{:}]==0)/2; 
hemispheres = [extracted.hemisphere{:}]; 
power_set = [10,17];

subjects = [extracted.subject{:}]; 
unique_subjects = unique(subjects);

powerSubjectComb  = combvec(1:numel(unique_subjects),power_set);

%%

for s=1:size(powerSubjectComb,2)
    subject = unique_subjects(powerSubjectComb(1,s)); 
    p = powerSubjectComb(2,s);
    
    for hem=-1:1
        ev = extracted.data{(subjects==subject) & (hemispheres==hem) & (powers==p)};
        
        plotOpt.toPlot=0; 
        
        c = get_rts(filterStructRows(ev,(ev.timeline_choiceMoveDir==1 & ev.is_laserTrial)),'rtMin',plotOpt) - ...
            get_rts(filterStructRows(ev,(ev.timeline_choiceMoveDir==1 & ~ev.is_laserTrial)),'rtMin',plotOpt); 
        
        i = get_rts(filterStructRows(ev,(ev.timeline_choiceMoveDir==2 & ev.is_laserTrial)),'rtMin',plotOpt) - ...
            get_rts(filterStructRows(ev,(ev.timeline_choiceMoveDir==2 & ~ev.is_laserTrial)),'rtMin',plotOpt);
        
        
        ipsi(s,hem+2) = nanmean(i,'all');
        contra(s,hem+2) =  nanmean(c,'all');


    end 
end 


%%
figure;
plot(ipsi','r');
hold on; 
plot(contra','b');
xticks([1,2,3])
xticklabels({'left','bi','right'})
ylabel('RTopto-RTcontrol')
legend({'left choice','right choice'})
hline(0,'k--')
%%