%%
clc; clear all; 
[sites,cortexDat] = loadCortexData('uni',1); 

% load the opto
extracted = loadOptoData('balanceTrials',0,'sepMice',0,'reExtract',1,'sepHemispheres',0,'sepPowers',0,'sepDiffPowers',0,'whichSet','uni_high'); 
%
sites{4} = 'SC'; 
allDat = [cortexDat,extracted.data]; 


shouldPlot = 1; 

plotfit = 1; % whether to connect the data or plot actual fits
plotParams.plottype = 'log'; 
% for each region get a delta fit for each subject.

%%
for site=1:numel(sites)
    currBlock = allDat{site};

    subjects = unique(currBlock.subjectID_);
    control_fit_params=ones(numel(subjects),6); 
    opto_fit_params=ones(numel(subjects),6); 
    for mouse=1:numel(subjects)
        optoBlock = filterStructRows(currBlock, currBlock.is_laserTrial & (currBlock.subjectID_==subjects(mouse))); 
        controlBlock = filterStructRows(currBlock, ~currBlock.is_laserTrial & (currBlock.subjectID_==subjects(mouse)));

        controlfit = plts.behaviour.GLMmulti(controlBlock, 'simpLogSplitVSplitA');
        controlfit.fit; 

        control_fit_params(mouse,:)= controlfit.prmFits; 

           
        if shouldPlot
            f=figure; 
            f.Position = [10,10,400,400];
            plotParams.LineStyle = '--';
            plotParams.DotStyle = 'none';
            plotParams.MarkerEdgeColor = 'k';
            plotParams.MarkerSize = 18; 
            plotParams.LineWidth = 3; 
            plotParams.addFake=0; 
    
            plot_optofit(controlfit,plotParams,plotfit)
        end 

        optoBlock.freeP  = logical([1,1,1,0,1,1]);
        orifit = plts.behaviour.GLMmulti(optoBlock, 'simpLogSplitVSplitA');
        orifit.prmInit = controlfit.prmFits;
        orifit.fitCV(5); 
        % how the parameters actually change 
        opto_fit_params(mouse,:) = mean(orifit.prmFits,1)+optoBlock.freeP.*controlfit.prmFits;

        if shouldPlot
           %
    	   orifit.prmFits(4) = controlfit.prmFits(4);
           plotParams.LineStyle = '-';
           plotParams.DotStyle = '.';
           plotParams.MarkerSize = 36; 
           plot_optofit(orifit,plotParams,plotfit,orifit.prmInit(4))
           title(sites{site})
        end


    end 

    ratios{site} = opto_fit_params./control_fit_params; 
    diffs{site} = opto_fit_params-control_fit_params;
    ctrlParams{site}=control_fit_params; 
    optoParams{site}=opto_fit_params; 


end 
%%
labels = {['bias'],['vis'],['aud']};
indices = {[1],[2,3],[5,6]};  


for p = 1:numel(labels)
    figure;
    for site=1:numel(sites)
        allratios = ratios{site}; 
        curr = mean(allratios(:,indices{p}),2); 
        plot(ones(numel(curr),1)*site+0.2*(site-2),curr,'ko')
        hold on
    end
    xticks([1,2,3,4])
    xticklabels(sites)
    title(labels{p})
    ylabel('opto/control param value')
    ylim([-1,4])
    hline(1,'--')
    xlim([.5,4.5])
end 

%%
labels = {['bias'],['V_i_p_s_i'],['V_c_o_n_t_r_a'],['A_i_p_s_i'],['A_c_o_n_t_r_a']};
indices = {[1],[2],[3],[5],[6]};  


for p = 1:numel(labels)
    figure;
    for site=1:numel(sites)
        allratios = ratios{site}; 
        curr = mean(allratios(:,indices{p}),2); 
        plot(ones(numel(curr),1)*site+0.2*(site-2),curr,'ko')
        hold on
    end
    xticks([1,2,3,4])
    xticklabels(sites)
    title(labels{p})
    ylabel('opto/control param value')
    %ylim([-1,2])
    hline(1,'--')
    xlim([.5,4.5])
end 

%%
labels = {['bias'],['V_i_p_s_i'],['V_c_o_n_t_r_a'],['A_i_p_s_i'],['A_c_o_n_t_r_a']};
indices = {[1],[2],[3],[5],[6]};  


for p = 1:numel(labels)
    figure;
    for site=1:numel(sites)
        allratios = diffs{site}; 
        curr = mean(allratios(:,indices{p}),2); 
        plot(ones(numel(curr),1)*site+0.2*(site-2),curr,'ko')
        hold on
    end
    xticks([1,2,3,4])
    xticklabels(sites)
    title(labels{p})
    ylabel('opto-control param value')
    ylim([-3,4])
    hline(0,'--')
    xlim([.5,4.5])
end 
%% plot the batch fits just for illustration

for site=1:numel(sites)
    currBlock = allDat{site};

    optoBlock = filterStructRows(currBlock, currBlock.is_laserTrial); 
    controlBlock = filterStructRows(currBlock, ~currBlock.is_laserTrial);

    controlfit = plts.behaviour.GLMmulti(controlBlock, 'simpLogSplitVSplitA');
    controlfit.fit; 


       
    if shouldPlot
        f=figure; 
        f.Position = [10,10,400,400];
        plotParams.LineStyle = '--';
        plotParams.DotStyle = 'none';
        plotParams.MarkerEdgeColor = 'k';
        plotParams.MarkerSize = 18; 
        plotParams.LineWidth = 3; 
        plotParams.addFake=0; 

        plot_optofit(controlfit,plotParams,plotfit)
    end 

    optoBlock.freeP  = logical([1,1,1,0,1,1]);
    orifit = plts.behaviour.GLMmulti(optoBlock, 'simpLogSplitVSplitA');
    orifit.prmInit = controlfit.prmFits;
    orifit.fitCV(5); 


    if shouldPlot
       %
	   orifit.prmFits(4) = controlfit.prmFits(4);
       plotParams.LineStyle = '-';
       plotParams.DotStyle = '.';
       plotParams.MarkerSize = 36; 
       plot_optofit(orifit,plotParams,plotfit,orifit.prmInit(4))
    end
   title(sites{site})
end 