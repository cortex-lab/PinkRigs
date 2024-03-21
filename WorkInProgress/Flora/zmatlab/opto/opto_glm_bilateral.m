clc; clear all;
extracted = loadOptoData('balanceTrials',0,'sepMice',1,'reExtract',1, ...
    'sepHemispheres',1,'sepPowers',1,'sepDiffPowers',1,'whichSet', 'bi_low'); 

%%
save_fig = 0;
savepath = 'D:\behaviours_opto'; 
%
% fit and plot each set of data
%
% fit sets that determine which parameters or combinations of parameters
% are allowed to change from fitting control trials to fitting opto trials
opto_fit_sets = logical([
    [0,0,0,0,0,0]; ... %1 
    [1,1,1,0,1,1]; ...
    [1,0,0,0,0,0]; ... %3 
    [1,1,1,0,0,0]; ... 
    [1,0,0,0,1,1]; ...
    [0,1,1,0,0,0]; ... %6
    [0,0,0,0,1,1]; ...
    [0,1,1,0,1,1]; ...   %8
    [1,0,0,0,1,1]; ...    
    [1,1,1,0,0,0]; ...  
    [0,0,0,0,1,1]; ...   %11
    [0,1,1,0,0,0]; ...   %12
]);

%%

plot_model_pred = zeros(size(opto_fit_sets,1),1); % indices of models to plot
plot_model_pred(2) = 1; 
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
        f=figure; 
        f.Position = [10,10,300,300];
        plotParams.LineStyle = '--';
        plotParams.DotStyle = 'none';
        plotParams.MarkerEdgeColor = 'k';
        plotParams.MarkerSize = 18; 
        plotParams.LineWidth = 3; 
        plotParams.addFake=1; 

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
%figure;  orifit.prmFits(4)
    	   orifit.prmFits(4) = controlfit.prmFits(4);
           plotParams.LineStyle = '-';
           plotParams.DotStyle = '.';
           plotParams.MarkerSize = 36; 
           plot_optofit(orifit,plotParams,plotfit,orifit.prmInit(4))
        end

    end

    %
    if save_fig
   
       savename = sprintf('%s_%.0fmW_hemisphere_%.0f_%splot.svg', ...
            extracted.subject{s},...
            extracted.power{s},...
            extracted.hemisphere{s},...
            plotParams.plottype); 
       saveas(gcf, [savepath '/' savename], 'svg');
    end 
    %
end


%%
% this stupid plot 
s_id = 1; 
figure; 

best_deltaR2 = opto_fit_logLik(s_id,1) - opto_fit_logLik(s_id,2);
yline(opto_fit_logLik(s_id,1),color='k',LineWidth=5)
hold on 
yline(opto_fit_logLik(s_id,2),color='green',LineWidth=5)
hold on 
plot([1,2,3],opto_fit_logLik(s_id,[6,7,3]),color='k',LineWidth=3,LineStyle='--'); % only 1 is allowed to change 
hold on 
plot([1,2,3],opto_fit_logLik(s_id,[9,10,8]),color='green',LineWidth=3,LineStyle='--'); % same one drops out
ylabel('-Log2Likelihood')
xticks([1,2,3])
xticklabels({'V','A','bias'})


sgtitle(sprintf('%s,hemisphere:%.0d,power:%.0d',extracted.subject{s_id},extracted.hemisphere{s_id},extracted.power{s_id}))





%
% this stupid plot 

% do the minmax normalisation for all parameters
figure;
sumR2 = opto_fit_logLik(:,1) - opto_fit_logLik(:,2);
minR2 = opto_fit_logLik(:,2);


opto_fit_logLik_norm = 1-((opto_fit_logLik-minR2)./sumR2); 

s_id = 3; 
plot([1,2,3],opto_fit_logLik_norm(s_id,[6,7,3]),color='k',LineWidth=3,LineStyle='-');
hold on 
plot([1,2,3],opto_fit_logLik_norm(s_id,[9,10,8]),color='green',LineWidth=3,LineStyle='-');
hold on 
sgtitle(sprintf('%s,hemisphere:%.0d,power:%.0d',extracted.subject{s_id},extracted.hemisphere{s_id},extracted.power{s_id}))
yline(0,'--')
yline(1,'--',color='k')

ylim([-0.05,1.05])
paramLabels = categorical({'V','A','bias'}); 
xticks([1,2,3,4,5])
xticklabels(paramLabels)
%hold on; 
%plot([1,2,3,4,5,6,7,8,9],mean(opto_fit_logLik_norm(:,[8,9,10,11,3,4,5,6,7])),color='k',LineWidth=5,LineStyle='-',Marker='+',MarkerSize=30);
%hold on; 
%plot([1,2,3,4,5,6,7,8,9],mean(opto_fit_logLik_norm(:,[13,14,15,16,12,17,18,19,20])),color='g',LineWidth=5,LineStyle='-',Marker='+',MarkerSize=30);
%%
% plot gain vs loss against each other for each param
paramLabels = categorical({'V','A','bias'}); 

gain_ids = [6,7,3]; 
loss_ids = [9,10,8];
figure; 
nSets = numel(gain_ids);
for i=1:nSets
    subplot(1,nSets,i)
    myx = opto_fit_logLik_norm(:,gain_ids(i));
    myy = 1-opto_fit_logLik_norm(:,loss_ids(i));
    plot(myx,myy,'.',MarkerSize=30,MarkerEdgeColor='cyan');hold on;
    yline(0); hold on;
    xline(0)
    plot([-.1;0],[1;1],color='k')
    plot([1;1],[-.1;0],color='k')

    xlim([-.1,1.1])
    ylim([-.1,1.1])
    [hv,p]= ttest(myx,myy);
    title(paramLabels(i),p)
end

%%

figure;
sumR2 = opto_fit_logLik(:,1) - opto_fit_logLik(:,2);
minR2 = opto_fit_logLik(:,2);


opto_fit_logLik_normBias = 1-((opto_fit_logLik-minR2)./sumR2); 

s_id = 1; 
for s_id=1:(size(opto_fit_logLik,1))
    hold on 
    plot([1,2,3,4],opto_fit_logLik_norm(s_id,[4,5,6,7])-opto_fit_logLik_norm(s_id,3),color='k',LineWidth=.1,LineStyle='-');
    %hold on 
    %plot([1,2,3,4],opto_fit_logLik_norm(s_id,[13,14,15,16]),color='green',LineWidth=1,LineStyle='-');
    %hold on 
    sgtitle(sprintf('%s,hemisphere:%.0d,power:%.0d',extracted.subject{s_id},extracted.hemisphere{s_id},extracted.power{s_id}))
end 
%ylim([-1,3])

%
%% 
% summary plot for how the each term changes between controlfit and full
% refit
    % fit

figure; 
paramLabels = categorical({'bias','V','A'}); 


newparams = cat(3,opto_fit_params(:,:,1), ...
            (opto_fit_params(:,:,2)+opto_fit_params(:,:,3))/2, ...
            (opto_fit_params(:,:,5)+opto_fit_params(:,:,6))/2);

for ptype=1:numel(paramLabels)
    subplot(1,numel(paramLabels),ptype)
    myx = newparams(:,1,ptype); 
    myy = newparams(:,1,ptype)+newparams(:,2,ptype);
    plot(myx,myy,'.',MarkerSize=30)
    hold on; 
    plot([-5,5],[-5,5],'k--')
    xlabel(sprintf('%s,control fit',paramLabels(ptype)))
    ylabel(sprintf('%s,full fit',paramLabels(ptype)))
    ylim([-5,5])
    [hv,p]= ttest(myx,myy);
    title(paramLabels(i),p)
    
end 

%%
% plot some parameters agains depth/cannula location in SC
locations = csv.readTable('D:\opto_cannula_locations.csv'); 

n = numel(extracted.data); 
dv = NaN(n,1); ap = NaN(n,1);ml = NaN(n,1);acronym = NaN(n,1); 
loc_subjects = cell2mat([locations.subject]);
loc_hemishpheres= str2double([locations.hemisphere{:}]);
% identify the location of each subject/hemishphere
for s=1:n 
    subject = extracted.subject{s};
    hemisphere = extracted.hemisphere{s}; 
    
    idx = find(locations.subject==subject & str2double(locations.hemisphere)==hemisphere);
    
    if numel(idx)==1 
        dv(s) = str2double(locations.dv{idx}); 
        ml(s) = str2double(locations.ml{idx}); 
        ap(s) = str2double(locations.ap{idx}); 


    end 
end 


%% % approximate stimulus 
ap_ccf = -ap+5400;
mid_loc = -3950; 
distance_from_stim = abs(ap_ccf-mid_loc); 


%
figure; 
paramLabels = categorical({'bias','Vipsi','Vcontra','gamma','Aipsi','Acontra'}); 

for ptype=1:numel(paramLabels)
    subplot(1,numel(paramLabels),ptype)

    plot(distance_from_stim,opto_fit_params(:,2,ptype),'.',MarkerSize=30)
    corrval = corrcoef(distance_from_stim,opto_fit_params(:,2,ptype), 'Rows','complete');
    hold on; 
    xlabel(sprintf('%s,ditance from stimulus',paramLabels(ptype)))
    ylabel(sprintf('%s,delta param, full refit',paramLabels(ptype)))
    if ptype==1
        ylim([-6,6])
    else
        ylim([-3,3])
    end 
    title(corrval(1,2))
    hold on; 
    yline(0)

end 

