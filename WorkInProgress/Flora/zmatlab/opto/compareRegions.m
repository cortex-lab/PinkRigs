%%
clc; clear all; close all;
 addpath(genpath('C:\Users\Flora\Documents\Github\PinkRigs'));
 addpath(genpath('C:\Users\Flora\Documents\Github\2023_CoenSit'));


set_type = 'bi';  
[sites,cortexDat] = loadCortexData(set_type,1); 

% load the opto
extracted = loadOptoData('balanceTrials',0,'sepMice',0,'reExtract',1,'sepHemispheres',0,'sepPowers',0,'sepDiffPowers',0,'whichSet',sprintf('%s_high',set_type)); 
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

if strcmp('bi',set_type)

    labels = {['bias'],['vis'],['aud']};
    indices = {[1],[2,3],[5,6]};  
elseif strcmp('uni',set_type)
    labels = {['bias'],['V_i_p_s_i'],['V_c_o_n_t_r_a'],['A_i_p_s_i'],['A_c_o_n_t_r_a']};
    indices = {[1],[2],[3],[5],[6]};  
end 


figure;
for p = 1:numel(labels)

    t = []; region_id = []; % for the anova

    subplot(1,numel(labels),p)
    for site=1:numel(sites)
        
        if strcmp('bias',labels{p})

            allratios = diffs{site}; 
            ylabel('opto - control param value')

        else 
            allratios = ratios{site}; 
            ylabel('opto/control param value')
            ylim([-1,4])

        end 

        param_per_site = mean(allratios(:,indices{p}),2); 
        
        % accumulate for ANOVA
        t = [t;param_per_site]; 
        region_id = [region_id;ones(numel(param_per_site),1)*site];

        % plot
        plot(ones(numel(param_per_site),1)*site+0.2*(site-2),param_per_site,'ko')
        hold on



    end
    % anova 
    allparams(p,:)=t;

    % plotting 
    xticks([1,2,3,4])
    xticklabels(sites)
    title(labels{p})
    hline(1,'--')
    xlim([.5,4.5])
end 

for i=1:numel(region_id)
    regions{i} = char(sites(region_id(i)));
end
%%

for p = 1:numel(labels)
    [p_anova, tbl_anova, stats_anova] = anova1(allparams(p,:),regions);
    
    % Display ANOVA results
    fprintf('ANOVA p-value for %s: %.4f\n', labels{p},p_anova);
    
    % If ANOVA is significant, perform post-hoc tests
    if p_anova < 0.05
        % Perform Tukey's HSD post-hoc test
        [c, m, h, gnames] = multcompare(stats_anova, 'CType', 'tukey-kramer');
        
        % Display post-hoc test results
        fprintf('Tukey''s HSD post-hoc test:\n');
        disp(array2table(c, 'VariableNames', {'Group1', 'Group2', 'LowerCI', 'UpperCI', 'Difference', 'PValue'}));
    end

end 

%%
labels = {['bias'],['V_i_p_s_i'],['V_c_o_n_t_r_a'],['A_i_p_s_i'],['A_c_o_n_t_r_a']};
indices = {[1],[2],[3],[5],[6]};  

figure;

for p = 1:numel(labels)
    subplot(1,numel(labels),p)
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
    if strcmp('bias',labels{p})
        ylim([-30,60])
    else
        ylim([-1,2])
    end
    %ylim([-1,2])
    hline(1,'--')
    xlim([.5,4.5])
end 



%% test performing the ANOVA with post-hoc

% basically I need to provide two vectosrs one value and one gouping


labels = {['bias'],['V_i_p_s_i'],['V_c_o_n_t_r_a'],['A_i_p_s_i'],['A_c_o_n_t_r_a']};
indices = {[1],[2],[3],[5],[6]};  
save_fig = 0; 
figure;
for p = 1:numel(labels)
    t = []; region_id = []; 
    for site=1:numel(sites)
        param_per_site = ratios{site}(:,indices{p});
        t = [t;param_per_site]; 
        region_id = [region_id;ones(numel(param_per_site),1)*site];
    end
    allparams(p,:)=t;  % the region id list is always the same
end 
for i=1:numel(region_id)
    regions{i} = char(sites(region_id(i)));
end


for p = 1:numel(labels)
    [p_anova, tbl_anova, stats_anova] = anova1(allparams(p,:),regions);
    
    % Display ANOVA results
    fprintf('ANOVA p-value for %s: %.4f\n', labels{p},p_anova);
    
    % If ANOVA is significant, perform post-hoc tests
    if p_anova < 0.1
        % Perform Tukey's HSD post-hoc test
        [c, m, h, gnames] = multcompare(stats_anova, 'CType', 'tukey-kramer');
        
        % Display post-hoc test results
        fprintf('Tukey''s HSD post-hoc test:\n');
        disp(array2table(c, 'VariableNames', {'Group1', 'Group2', 'LowerCI', 'UpperCI', 'Difference', 'PValue'}));
    end

end 


%%
labels = {['bias'],['V_i_p_s_i'],['V_c_o_n_t_r_a'],['A_i_p_s_i'],['A_c_o_n_t_r_a']};
indices = {[1],[2],[3],[5],[6]};  
save_fig = 0; 
figure;
for p = 1:numel(labels)
    subplot(1,numel(labels),p);
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


%% test performing the ANOVA with post-hoc

% basically I need to provide two vectosrs one value and one gouping


labels = {['bias'],['V_i_p_s_i'],['V_c_o_n_t_r_a'],['A_i_p_s_i'],['A_c_o_n_t_r_a']};
indices = {[1],[2],[3],[5],[6]};  
save_fig = 0; 
figure;
for p = 1:numel(labels)
    t = []; region_id = []; 
    for site=1:numel(sites)
        param_per_site = diffs{site}(:,indices{p});
        t = [t;param_per_site]; 
        region_id = [region_id;ones(numel(param_per_site),1)*site];
    end
    allparams(p,:)=t;  % the region id list is always the same
end 
for i=1:numel(region_id)
    regions{i} = char(sites(region_id(i)));
end



for p = 1:numel(labels)
    [p_anova, tbl_anova, stats_anova] = anova1(allparams(p,:),regions);
    
    % Display ANOVA results
    fprintf('ANOVA p-value for %s: %.4f\n', labels{p},p_anova);
    
    % If ANOVA is significant, perform post-hoc tests
    if p_anova < 0.1
        % Perform Tukey's HSD post-hoc test
        [c, m, h, gnames] = multcompare(stats_anova, 'CType', 'tukey-kramer');
        
        % Display post-hoc test results
        fprintf('Tukey''s HSD post-hoc test:\n');
        disp(array2table(c, 'VariableNames', {'Group1', 'Group2', 'LowerCI', 'UpperCI', 'Difference', 'PValue'}));
    end

end 


%% plot the batch fits just for illustration
save_fig=1; savepath = 'D:\behaviours_opto'; 

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

    if save_fig
   
       savename = sprintf('%s_',sites{site}); 
       saveas(gcf, [savepath '/' savename], 'svg');
    end 
end 