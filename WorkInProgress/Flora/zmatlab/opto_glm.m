clc; clear all;
extracted = loadOptoData('balanceTrials',0,'sepMice',1,'reExtract',1,'sepHemispheres',1); 

%
% fit and plot each set of data
%
% fit sets that determine which parameters or combinations of parameters
% are allowed to change from fitting control trials to fitting opto trials
opto_fit_sets = logical([
    [0,0,0,0,0,0]; ... %1 
    [1,1,1,0,1,1]; ...
    [1,0,0,0,0,0]; ...
    [1,1,0,0,0,0]; ... %4
    [1,0,1,0,0,0]; ... 
    [1,0,0,0,1,0]; ...
    [1,0,0,0,0,1]; ...
    [0,1,0,0,0,0]; ... %8
    [0,0,1,0,0,0]; ...
    [0,0,0,0,1,0]; ...
    [0,0,0,0,0,1]; ...
    [0,1,1,0,1,1]; ...   %12
    [1,0,1,0,1,1]; ...    
    [1,1,0,0,1,1]; ...    
    [1,1,1,0,0,1]; ...    
    [1,1,1,0,1,0]; ...    
]);

%%

plot_model_pred = zeros(size(opto_fit_sets,1),1); % indices of models to plot
plot_model_pred(3) = 1; 
shouldPlot = 1; 

plotfit = 1; % whether to connect the data or plot actual fits
plotParams.plottype = 'log'; 
for s=1:numel(extracted.data)    
    currBlock = extracted.data{s};
    nTrials(s) = numel(currBlock.is_blankTrial); 
    optoBlock = filterStructRows(currBlock, currBlock.is_laserTrial); 
    controlBlock = filterStructRows(currBlock, ~currBlock.is_laserTrial);

    % 
    %optoBlock = addFakeTrials(optoBlock);
    %controlBlock = addFakeTrials(controlBlock);

    % fit and plot
    controlfit = plts.behaviour.GLMmulti(controlBlock, 'simpLogSplitVSplitA');
    controlfit.fit; 
    control_fit_params(s,:)= controlfit.prmFits; 
    
    if shouldPlot
        figure; 
        plotParams.LineStyle = '-';
        plotParams.DotStyle = '.';
        plotParams.MarkerSize = 24; 
        plotParams.LineWidth = 3; 

        plot_optofit(controlfit,plotParams,plotfit)
        hold on; 
        title(sprintf('%s,%.0d opto,%.0d control trials, %.0f mW, %.0f', ...
            extracted.subject{s},...
            numel(optoBlock.is_blankTrial),... 
            numel(controlBlock.is_blankTrial),...
            extracted.power{s},...
            extracted.hemisphere{s}))
    end


    %
    for model_idx = 1:size(opto_fit_sets,1)
        optoBlock.freeP  = opto_fit_sets(model_idx,:);
        orifit = plts.behaviour.GLMmulti(optoBlock, 'simpLogSplitVSplitA');
        orifit.prmInit = controlfit.prmFits;
        orifit.fitCV(5); 
        opto_fit_logLik(s,model_idx) = mean(orifit.logLik);
        % how the parameters actually change 
        opto_fit_params(s,model_idx,:) = mean(orifit.prmFits,1);

        if shouldPlot && plot_model_pred(model_idx)
           %
% figure;  orifit.prmFits(4)
    	   %orifit.prmFits(4) = controlfit.prmFits(4);
           plotParams.LineStyle = '--';
           plotParams.DotStyle = '.';
           plotParams.MarkerSize = 24; 
           plot_optofit(orifit,plotParams,plotfit,orifit.prmInit(4))
        end

    end
    %
end

%%
% summary plots for cross-validation 
% only include things that are more than 2k trials
paramLabels = categorical({'bias','Vipsi','Vcontra','Aipsi','Acontra'}); 

% normalise the log2likelihood


 

% plot order; % calculate fit improvement by varying each predictor 

best_deltaR2 = opto_fit_logLik(:,1) - opto_fit_logLik(:,2);
deltaR2 = (opto_fit_logLik(:,1)-opto_fit_logLik(:,[3,8,9,10,11]))./best_deltaR2;


cvR2 = (opto_fit_logLik(:,2)-opto_fit_logLik(:,12:16))./best_deltaR2;

%% individual plots 
figure; 
plot(deltaR2');

figure;plot(cvR2'); 

%% plot bar plot of summary 
figure;
errorbar(paramLabels,median(deltaR2),zeros(size(deltaR2,2),1),mad(deltaR2),'black',"LineStyle","none");
hold on; 
bar(paramLabels,median(deltaR2),'black');
hold on; 
errorbar(paramLabels,median(cvR2),mad(cvR2),zeros(size(deltaR2,2),1),'green',"LineStyle","none");
hold on;
bar(paramLabels,median(cvR2),'green'); 

%% 
% compare the actual values for each mice 

figure; 


plot([1,2],median(opto_fit_logLik(:,2:3)),['black']);
hold on
for m=1:size(opto_fit_logLik,1)
    plot([1,2],[opto_fit_logLik(m,2),opto_fit_logLik(m,3)],'k')
    hold on 
end 
[h,p]= ttest(opto_fit_logLik(:,2),opto_fit_logLik(:,3));

%%
labels = categorical({'bias','Vipsi','Vcontra','Aipsi','Acontra'}); 

figure;
bar([1,2,3,4,5],median(opto_fit_logLik(:,3:7)),['black']);
hold on
for m=1:size(opto_fit_logLik,1)
    plot([1,2,3,4,5],[opto_fit_logLik(m,3:7)])
    hold on 
end 


%% 
% summary plot for how the each term changes between controlfit and full
% refit
    % fit

figure; 
for ptype=1:numel(paramLabels)
    subplot(1,numel(paramLabels),ptype)
    plot(opto_fit_params(:,1,ptype),opto_fit_params(:,1,ptype)+opto_fit_params(:,2,ptype),'o')
    hold on; 
    plot([-5,5],[-5,5],'k--')
    xlabel(sprintf('%s,control fit',paramLabels(ptype)))
    ylabel(sprintf('%s,full fit',paramLabels(ptype)))
    ylim([-5,5])
end 

%%
% plot some parameters agains depth/cannula location in SC
locations = csv.readTable('C:\Users\Flora\Documents\Processed data\Audiovisual\cannula_locations.csv'); 

n = numel(extracted.data); 
dv = NaN(n,1); acronym = NaN(n,1); 
loc_subjects = cell2mat([locations.subject]);
loc_hemishpheres= str2double([locations.hemisphere{:}]);
% identify the location of each subject/hemishphere
for s=1:n 
    subject = extracted.subject{s};
    hemisphere = extracted.hemisphere{s}; 
    
    idx = find(locations.subject==subject & str2double(locations.hemisphere)==hemisphere);
    
    if numel(idx)==1 
        dv(s) = str2double(locations.dv{idx}); 
    end 
end 

%%
% plot the dv location against the 
figure; plot(deltaR2(:,3),dv,'o'); 
