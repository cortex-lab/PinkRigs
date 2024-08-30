clc; clear all; 
d = readtable('D:\LogRegression\opto\uni_all_nogo\formatted\fit_results\summary.csv'); 

%%

figure; 
paramLabels = ({'visR','visL','audR','audL','bias'}); 

which = 'distance from stimulus'; 

%
which = 'eYFP exxpression';

for ptype=1:numel(paramLabels)
    name = char((paramLabels(ptype)));
    deltaname = sprintf('delta_%s',name); 

    subplot(1,numel(paramLabels),ptype)
    
    if strcmp('distance from stimulus',which)        
        myx =d.distance_from_stim; 
    elseif strcmp('eYFP exxpression',which)
        myx =d.eYFP_fluorescence; 
    end 

    myy = d.(deltaname);

    tbl = table; 
    tbl.distance = d.distance_from_stim; 
    tbl.paramchange = myy; 


    tbl.eyfp = d.eYFP_fluorescence;
    tbl.hemisphere = categorical(d.hemisphere); 
    tbl.subject = categorical(d.subject); 

    tbl = tbl(~isnan(myx),:);
    %model = fitlme(tbl, 'paramchange ~ distance + (1|subject) + (1|hemisphere) + (1|eyfp)');
    model = fitlme(tbl, 'paramchange ~ eyfp + (1|subject) + (1|hemisphere) + (1|distance)');
    % Get the coefficients table
    coeff_table = model.Coefficients;    
    % Find the row index corresponding to the 'Group' predictor
    group_idx = find(strcmp(coeff_table.Name, 'eyfp'));    
    % Extract the p-value
    p_value = coeff_table.pValue(group_idx);

    ax= plot(myx,myy,'.',MarkerSize=30);
    hold on; 
    xlabel(sprintf('%s,%s',name,which))
    if strcmp('bias',name)
        ylabel(sprintf('opto-control param, full refit'))
        yline(0)
        ylim([-6,6])
    else
        ylim([0,2])
        yline(1)
        ylabel(sprintf('opto/cotrol param, full refit'))
        %yscale(ax,'log')


    end 
    title(name,p_value)
    hold on; 

% end 
end 



%%
% 
%     if strcmp('bias',string(paramLabels(ptype)))
%         myy = (opto_fit_params(:,2,ptype));
%     else
%         myy = (opto_fit_params(:,2,ptype)+control_fit_params(:,ptype))./control_fit_params(:,ptype);
%     end 
% 
%     tbl = table; 
%     tbl.distance = myx; 
%     tbl.paramchange = myy; 
% 
%     tbl.eyfp = eyfp;
%     tbl.hemisphere = categorical([extracted.hemisphere{:}])'; 
%     tbl.subject = categorical([extracted.subject{:}])'; 
% 
%     tbl = tbl(~isnan(myx),:);
%     model = fitlme(tbl, 'paramchange ~ distance + (1|subject) + (1|hemisphere)');
%     % Get the coefficients table
%     coeff_table = model.Coefficients;    
%     % Find the row index corresponding to the 'Group' predictor
%     group_idx = find(strcmp(coeff_table.Name, 'distance'));    
%     % Extract the p-value
%     p_value = coeff_table.pValue(group_idx);
% 
%     ax= plot(myx,myy,'.',MarkerSize=30);
%     hold on; 
%     xlabel(sprintf('%s,%s',paramLabels(ptype),which))
%     if strcmp('bias',string(paramLabels(ptype)))
%         ylabel(sprintf('opto-control param, full refit'))
%         yline(0)
%         ylim([-6,6])
%     else
%         ylim([-3,3])
%         yline(1)
%         ylabel(sprintf('opto/cotrol param, full refit'))
%         %yscale(ax,'log')
% 
% 
%     end 
%     title(paramLabels(ptype),p_value)
%     hold on; 
% 
% end 