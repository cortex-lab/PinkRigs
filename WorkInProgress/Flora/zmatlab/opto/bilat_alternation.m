clc; clear all;
extracted = loadOptoData('balanceTrials',0,'sepMice',1,'reExtract',1,'sepHemispheres',1,'whichSet', 'bi_alternate');

%%

 plotOpt.toPlot=0; 

for s=1:numel(extracted.subject)    
    ev = extracted.data{s};


    dpR(s) = nanmean(get_metrics_StimClass(filterStructRows(ev,(ev.is_laserTrial)),'pR',plotOpt) - ...
                         get_metrics_StimClass(filterStructRows(ev,(~ev.is_laserTrial)),'pR',plotOpt),'all');

    postlaser(s) = nanmean(get_metrics_StimClass(filterStructRows(ev,(ev.postLaserTrial & ~ev.is_laserTrial)),'pR',plotOpt) - ...
                         get_metrics_StimClass(filterStructRows(ev,(~ev.postLaserTrial & ~ev.is_laserTrial)),'pR',plotOpt),'all');
end 
%


 %%
powers = [extracted.power{:}]; 
powers([extracted.hemisphere{:}]==0) = powers([extracted.hemisphere{:}]==0)/2; 
hemispheres = [extracted.hemisphere{:}]; 
power_set = [17];

subjects = [extracted.subject{:}]; 

unique_subjects = unique(subjects);

powerSubjectComb  = combvec(1:numel(unique_subjects),power_set);

%%


hemispheres = unique([extracted.hemisphere{:}]);


for h=1:numel(hemispheres)
    
end 