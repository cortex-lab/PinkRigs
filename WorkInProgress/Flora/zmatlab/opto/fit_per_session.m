% fit every session separately, per subject, per power, per hemisphere
clc; clear all;
extracted = loadOptoData('balanceTrials',0,'sepMice',1,'reExtract',0,'sepHemispheres',1,'sepPowers',1,'sepDiffPowers',1); 

% 
plotParams.LineStyle = '--';
plotParams.DotStyle = '.';
plotParams.MarkerEdgeColor = 'k';
plotParams.MarkerSize = 18; 
plotParams.LineWidth = 3; 
plotParams.addFake=1; 
plotfit = 1; % whether to connect the data or plot actual fits
plotParams.plottype = 'sig'; 

for s=1:numel(extracted.data)    
    currBlock = extracted.data{s};
    sessions = unique(currBlock.sessionID);

    for cs=1:numel(sessions)
        optoBlock = filterStructRows(cur rBlock, (currBlock.is_laserTrial) & (currBlock.sessionID==sessions(cs))); 
        controlBlock = filterStructRows(currBlock, (~currBlock.is_laserTrial) & (currBlock.sessionID==sessions(cs)));


         nTrialsCtrl(s,cs) = numel(controlBlock.is_blankTrial); 

         nTrials(s,cs) = numel(optoBlock.is_blankTrial); 

        % fit and plot
    	controlfit = plts.behaviour.GLMmulti(controlBlock, 'simpLogSplitVSplitA');
        controlfit.fit; 
        control_fit_params(s,cs,:)= controlfit.prmFits; 


        optoBlock.freeP  = logical([1,1,1,0,1,1]);
        orifit = plts.behaviour.GLMmulti(optoBlock, 'simpLogSplitVSplitA');
        orifit.prmInit = controlfit.prmFits;
        orifit.fit; 
        % the new parameters
        newP = orifit.prmInit+optoBlock.freeP.*orifit.prmFits;
        opto_fit_params(s,cs,:) = newP; 
        f=figure; 
        f.Position = [10,10,400,400];


        plot_optofit(orifit,plotParams,plotfit)
        title(sprintf('n_trials=%.0f, b:%.1f,V_i:%.1f,\n V_c:%.1f, g:%.2f,A_i:%.1f,A_c:%.1f',[nTrials(s,cs),newP]))
        xlabel('contrast')
        ylabel('pR')
        ylim([0,1])
    end 
end 
%%

% plot param changes on sessions
sel = 'bias';
labels = categorical({'bias','Vipsi','Vcontra','Aipsi','Acontra'});
pidxs = [1,2,3,5,6]; 
minTrial = 100; 
figure;
colors = { 'magenta','k', 'cyan'};

cids = [2,4,6]; 
for param=1:nPlots
    figure;
    nPlots = numel(labels);
    subplot(1,nPlots,param)
    for s_id=1:numel(cids) 
        figure;
        cnTrial = nTrials(s_id,:)>minTrial;
        plot([control_fit_params(cids(s_id),cnTrial,pidxs(param));opto_fit_params(cids(s_id),cnTrial,pidxs(param))],color=colors{s_id})
        
        
        %[h(param),p(param)]= ttest(control_fit_params(s_id,cnTrial,pidxs(param)),opto_fit_params(s_id,cnTrial,pidxs(param)));
        hold on
        title(labels(param))
       % title(sprintf('%s,p=%.3f',labels(param),p(param)));
        %sgtitle(sprintf('%s,hemisphere:%.0d,power:%.0d',extracted.subject{s_id},extracted.hemisphere{s_id},extracted.power{s_id}))

        xticks([1,2])
        xticklabels({'ctrl','opto'})
        %set(gca, 'YScale', 'log')
    end 
end

%% 
% separate plot per mouse
cids = [1,2,3,4,5,6]; 

for s_id=1:numel(cids) 
      figure;

    for param=1:nPlots
        nPlots = numel(labels);
        subplot(1,nPlots,param)
        cnTrial = nTrials(s_id,:)>minTrial;
        plot([control_fit_params(cids(s_id),cnTrial,pidxs(param));opto_fit_params(cids(s_id),cnTrial,pidxs(param))],color='k')
        
        
        [h(param),p(param)]= ttest(control_fit_params(s_id,cnTrial,pidxs(param)),opto_fit_params(s_id,cnTrial,pidxs(param)));
        hold on
        title(labels(param))
        title(sprintf('%s,p=%.3f',labels(param),p(param)));
        sgtitle(sprintf('%s,hemisphere:%.0d,power:%.0d',extracted.subject{s_id},extracted.hemisphere{s_id},extracted.power{s_id}))

        xticks([1,2])
        xticklabels({'ctrl','opto'})
        %set(gca, 'YScale', 'log')
    end 
end

%%
% plot all mice together 
figure;
nPlots = numel(labels);
for param=1:nPlots
    subplot(1,nPlots,param)

    a = control_fit_params(:,cnTrial,pidxs(param)); 
    b = opto_fit_params(:,cnTrial,pidxs(param));
    a = a(:); b=b(:); 
	plot([a,b]',color='k')
    [h(param),p(param)]= ttest(a,b);
    title(sprintf('%s,p=%.3f',labels(param),p(param)));

    xticks([1,2])
    xticklabels({'ctrl','opto'})
    set(gca, 'YScale', 'log')

end 

%%
figure; 
plot(control_fit_params(:,:,1),opto_fit_params(:,:,5),'.'); 


%% is this trial type
param = 1;

biases_c = control_fit_params(:,:,param); 
biases_o = opto_fit_params(:,:,param); 
minT = 0;
figure; plot(nTrials(nTrials>minT),biases_o(nTrials>minT),'.',color='k');

hold on; 

plot(nTrials(nTrials>minT),biases_c(nTrials>minT),'.',color='r');

xlabel('no. of trials')
ylabel(labels(param))
legend({'opto';'control'})

