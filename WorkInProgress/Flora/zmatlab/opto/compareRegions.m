%%
clc; clear all; close all;
 addpath(genpath('C:\Users\Flora\Documents\Github\PinkRigs'));
 addpath(genpath('C:\Users\Flora\Documents\Github\2023_CoenSit'));


set_type = 'uni';  
[sites,cortexDat] = loadCortexData(set_type,1); 

% load the opto
extracted = loadOptoData('balanceTrials',0,'sepMice',0,'reExtract',1,'sepHemispheres',0,'sepPowers',0,'sepDiffPowers',0,'whichSet',sprintf('%s_all',set_type)); 
%
sites{4} = 'SC'; 
allDat = [cortexDat,extracted.data]; 



%%

% Desired new order for the sites
new_order = {'SC','Frontal', 'Vis', 'Lateral'};

% Initialize new data cell array
new_data = cell(1, 4);
% Loop through the new order and rearrange the data accordingly
for i = 1:length(new_order)
    % Find the index in the original 'sites' cell that matches the current site in 'new_order'
    idx = find(strcmp(sites, new_order{i}));
    
    % Assign the corresponding data to the new data cell
    new_data{i} = allDat{idx};
end

% Update the original sites and data with the new order
sites = new_order';
allDat = new_data;




 %%
 % save out the data!


basefolder = ['D:\LogRegression\opto', '\', 'region_comparison', '\' , set_type]; 

if ~exist(basefolder, 'dir')
    mkdir(basefolder);
end
%
for i=1:numel(allDat)
    namestring = sprintf('%s.csv',sites{i});
    table = struct2table(allDat{i});
    csv.writeTable(table,[basefolder,'\',namestring])
end




%%
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



f=figure; 
f.Position = [10,10,numel(labels)*200,200];
for p = 1:numel(labels)

    t = []; region_id = []; % for the anova

    subplot(1,numel(labels),p)
    % plotting 


    for site=1:numel(sites)
        
        if strcmp('bias',labels{p})

            allratios = diffs{site}; 
            ylabel('opto - control param value')
            ylim([-1,5])


        else 
            allratios = ratios{site}; 
            ylabel('opto/control param value')
            ylim([-1,2])


        end 

        param_per_site = mean(allratios(:,indices{p}),2); 
        
        % accumulate for ANOVA
        t = [t;param_per_site]; 
        region_id = [region_id;ones(numel(param_per_site),1)*site];

        % plot
        plot(ones(numel(param_per_site),1)*site,param_per_site,'k.','MarkerSize',30)
        hold on

        if strcmp('bias',labels{p})
            hline(0,'k--')
        else
            hline(1,'k--')
        end 


    end
    % anova 
    allparams(p,:)=t;

    title(labels{p})

    xticks([1,2,3,4])
    xticklabels(sites)
    xlim([0,4.2])


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

%% plot the batch fits just for illustration
save_fig=1; savepath = 'D:\behaviours_opto'; 

for site=1:numel(sites)
    currBlock = allDat{site};

    if strcmp('bi',set_type)
        currBlock.stim_visDiff =  round(currBlock.stim_visDiff,1);
    end 

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
   
       savename = sprintf('%s_%s',sites{site},set_type); 
       saveas(gcf, [savepath '/' savename], 'svg');
    end 
end 