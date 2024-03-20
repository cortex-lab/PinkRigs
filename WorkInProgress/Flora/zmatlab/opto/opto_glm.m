clc; clear all;
extracted = loadOptoData('balanceTrials',0,'sepMice',1,'reExtract',1, ...
    'sepHemispheres',1,'sepPowers',1,'sepDiffPowers',0,'whichSet', 'uni_all'); 


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
    [0,0,1,0,1,1]; ...   %17
    [0,1,0,0,1,1]; ...   %18
    [0,1,1,0,0,1]; ...   %19
    [0,1,1,0,1,0]; ...   %20

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
% compare bias vs bias+visContra


figure; 


%
%plot([1,2],median(opto_fit_logLik(:,2:3)),['black']);
hold on
for m=1:size(opto_fit_logLik,1)
    plot([1,2],[opto_fit_logLik(m,3),opto_fit_logLik(m,5)],'k')
    hold on 
end 
[hv,pv]= ttest(opto_fit_logLik(:,3),opto_fit_logLik(:,5));

%%


deltaVVb = opto_fit_logLik(:,3)-opto_fit_logLik(:,5);
%%
% this stupid plot 
s_id = 2; 
figure; 

best_deltaR2 = opto_fit_logLik(s_id,1) - opto_fit_logLik(s_id,2);
ax1 = subplot(1,2,1);
yline(opto_fit_logLik(s_id,1),color='k',LineWidth=5)
hold on 
yline(opto_fit_logLik(s_id,2),color='green',LineWidth=5)
hold on 
plot([1,2,3,4,5],opto_fit_logLik(s_id,[8,9,10,11,3]),color='k',LineWidth=3,LineStyle='--'); % only 1 is allowed to change 
hold on 
plot([1,2,3,4,5],opto_fit_logLik(s_id,[13,14,15,16,12]),color='green',LineWidth=3,LineStyle='--'); % same one drops out
ylabel('-Log2Likelihood')
xticks([1,2,3,4,5])
xticklabels({'V_i_p_s_i','V_c_o_n_t_r_a','A_i_p_s_i','A_c_o_n_t_r_a','bias'})

ax2 = subplot(1,2,2);
yline(opto_fit_logLik(s_id,3),color='k',LineWidth=5) % bias add 
hold on 
yline(opto_fit_logLik(s_id,12),color='green',LineWidth=5) % bias lost 
plot([1,2,3,4],opto_fit_logLik(s_id,[4:7]),color='k',LineWidth=3,LineStyle='--'); % bias + x how much more it gains 
hold on 
plot([1,2,3,4],opto_fit_logLik(s_id,[17:20]),color='green',LineWidth=3,LineStyle='--',Marker='+'); % bias + x how much more it looses 
linkaxes([ax1, ax2], 'y');
xticks([1,2,3,4])
xticklabels({'V_i_p_s_i','V_c_o_n_t_r_a','A_i_p_s_i','A_c_o_n_t_r_a'})


sgtitle(sprintf('%s,hemisphere:%.0d,power:%.0d',extracted.subject{s_id},extracted.hemisphere{s_id},extracted.power{s_id}))





%%
% this stupid plot 

% do the minmax normalisation for all parameters
figure;
sumR2 = opto_fit_logLik(:,1) - opto_fit_logLik(:,2);
minR2 = opto_fit_logLik(:,2);


opto_fit_logLik_norm = 1-((opto_fit_logLik-minR2)./sumR2); 

s_id = 6; 
plot([1,2,3,4,5],opto_fit_logLik_norm(s_id,[8,9,10,11,3]),color='k',LineWidth=3,LineStyle='-');
hold on 
plot([1,2,3,4,5],opto_fit_logLik_norm(s_id,[13,14,15,16,12]),color='green',LineWidth=3,LineStyle='-');
hold on 
sgtitle(sprintf('%s,hemisphere:%.0d,power:%.0d',extracted.subject{s_id},extracted.hemisphere{s_id},extracted.power{s_id}))
yline(0,'--')
yline(1,'--',color='k')

ylim([-0.05,1.05])
paramLabels = categorical({'Vipsi','Vcontra','Aipsi','Acontra','bias'}); 
xticks([1,2,3,4,5])
xticklabels(paramLabels)
%hold on; 
%plot([1,2,3,4,5,6,7,8,9],mean(opto_fit_logLik_norm(:,[8,9,10,11,3,4,5,6,7])),color='k',LineWidth=5,LineStyle='-',Marker='+',MarkerSize=30);
%hold on; 
%plot([1,2,3,4,5,6,7,8,9],mean(opto_fit_logLik_norm(:,[13,14,15,16,12,17,18,19,20])),color='g',LineWidth=5,LineStyle='-',Marker='+',MarkerSize=30);
%%
% plot gain vs loss against each other for each param
paramLabels = categorical({'Vipsi','Vcontra','Aipsi','Acontra','bias'}); 

gain_ids = [8,9,10,11,3]; 
loss_ids = [13,14,15,16,12];
figure; 
nSets = numel(gain_ids);
for i=1:nSets
    subplot(1,nSets,i)
    plot(opto_fit_logLik_norm(:,gain_ids(i)),1-opto_fit_logLik_norm(:,loss_ids(i)),'.',MarkerSize=30,MarkerEdgeColor='cyan');hold on;
    yline(0); hold on;
    xline(0)
    plot([-.1;0],[1;1],color='k')
    plot([1;1],[-.1;0],color='k')

    xlim([-.1,1.1])
    ylim([-.1,1.1])
    title(paramLabels(i))
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

%%
gain_ids = [4,5,6,7]; 
loss_ids = [13,14,15,16];
figure; 
nSets = numel(gain_ids);
for i=1:nSets
    subplot(1,nSets,i)
    plot(opto_fit_logLik_norm(:,gain_ids(i))-opto_fit_logLik_norm(:,3),1-opto_fit_logLik_norm(:,loss_ids(i)),'.',MarkerSize=30);hold on;
    yline(0); hold on;
    xline(0)
    xlim([-.06,.06])
    ylim([-.06,.06])
end

%%
gain_ids = [4,5,6,7]; 
i=4;
[(opto_fit_logLik_norm(:,gain_ids(i))-opto_fit_logLik_norm(:,3)),(1-opto_fit_logLik_norm(:,loss_ids(i)))]
%%
s_id=2;

sprintf('%s,hemisphere:%.0d,power:%.0d',extracted.subject{s_id},extracted.hemisphere{s_id},extracted.power{s_id})
%% 
% summary plot for how the each term changes between controlfit and full
% refit
    % fit

figure; 
paramLabels = categorical({'bias','Vipsi','Vcontra','gamma','Aipsi','Acontra'}); 
subjects = categorical([extracted.subject{:}])';
for ptype=1:numel(paramLabels)
    subplot(1,numel(paramLabels),ptype)
    myx = opto_fit_params(:,1,ptype);
    myy = opto_fit_params(:,1,ptype)+opto_fit_params(:,2,ptype);

     for s =1:numel(subjects)
        myx_hemisaveraged(s) = mean(myx(subjects==subjects(s)));
        myy_hemisaveraged(s) = mean(myy(subjects==subjects(s)));

     end 

    plot(myx,myy,'.',MarkerSize=30)
    
    hold on; 
    plot([-5,5],[-5,5],'k--')
    xlabel(sprintf('%s,control fit',paramLabels(ptype)))
    ylabel(sprintf('%s,full fit',paramLabels(ptype)))
    ylim([-5,5])
    [hv,p]= ttest(myx_hemisaveraged,myy_hemisaveraged);
    title(p)
end 

%


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

% gain likelihood 8,9,10,11,3 'Vipsi','Vcontra','Aipsi','Acontra','bias'
% loss likelihood 13,14,15,16,12
% param change

idx_sets = [3,12,1];  % bias
idx_sets = [9,14,3];  % Vcontra
idx_sets = [8,13,2];  % Vipsi
idx_sets = [11,16,6];  % A contra
idx_sets = [10,15,5];  % Aipsi

gainll = opto_fit_logLik_norm(:,idx_sets(1)); 
lossll = 1-opto_fit_logLik_norm(:,idx_sets(2)); 
paramchange = opto_fit_params(:,2,idx_sets(3));
figure;
subplot(1,3,1)
plot(distance_from_stim,gainll,'.',Markersize=30);
subplot(1,3,2)
plot(distance_from_stim,lossll,'.',Markersize=30);

subplot(1,3,3)
plot(distance_from_stim,paramchange,'.',Markersize=30);


%%
figure; 
paramLabels = categorical({'bias','Vipsi','Vcontra','gamma','Aipsi','Acontra'}); 

for ptype=1:numel(paramLabels)
    subplot(1,numel(paramLabels),ptype)

    myx =distance_from_stim; 

    if strcmp('bias',paramLabels(ptype))
        myy = (opto_fit_params(:,2,ptype));
    else
        myy = (opto_fit_params(:,2,ptype)+opto_fit_params(:,1,ptype))./opto_fit_params(:,1,ptype);
    end 

    tbl = table; 
    tbl.distance = myx; 
    tbl.paramchange = myy; 
    tbl.hemisphere = categorical([extracted.hemisphere{:}])'; 
    tbl.subject = categorical([extracted.subject{:}])'; 

    tbl = tbl(~isnan(myx),:);
    model = fitlme(tbl, 'paramchange ~ distance + (1|subject) + (1|hemisphere)');
    % Get the coefficients table
    coeff_table = model.Coefficients;    
    % Find the row index corresponding to the 'Group' predictor
    group_idx = find(strcmp(coeff_table.Name, 'distance'));    
    % Extract the p-value
    p_value = coeff_table.pValue(group_idx);

    plot(myx,myy,'.',MarkerSize=30)
    hold on; 
    xlabel(sprintf('%s,ditance from stimulus',paramLabels(ptype)))
    if ptype==1
        ylabel(sprintf('opto-cotrol param, full refit'))
        yline(0)
        ylim([-6,6])
    else
        ylim([-3,3])
        yline(1)
        ylabel(sprintf('opto/cotrol param, full refit'))
        yscale log


    end 
    title(paramLabels(ptype),p_value)
    hold on; 

end 

