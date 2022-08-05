% before after implant plots 
close all;
subjects = {'AV014','AV015','AV005','AV006','AV008','AV007','FT030','FT031','FT032','FT035','AV020','AV021'}; 
%subjects = {'AV015'}; 
mainCSVLoc = csv.getLocation('main');
mainCSV = csv.readTable(mainCSVLoc);

for m_idx=1:numel(subjects)
    is_subject = cellfun(@(x) strcmp(x, subjects{m_idx}),mainCSV.Subject);
    implant_date = mainCSV(is_subject,:).P0_implantDate{1};
    
    
    before_string = [datestr(daysadd(implant_date,-10), 'yyyy-mm-dd'),':',implant_date];
    
    after_string = [datestr(daysadd(implant_date,1), 'yyyy-mm-dd'),':',datestr(daysadd(implant_date,15), 'yyyy-mm-dd')];
    %plt.behaviour.boxPlots('subject',subjects{m_idx},'expDate',before_string,'sepPlots',0);
    %plt.behaviour.boxPlots('subject',subjects{m_idx},'expDate',after_string,'sepPlots',0);
    
    [meanvis_before(m_idx),meanaud_before(m_idx)]  = get_data('subject',subjects{m_idx},'expDate',before_string); 
    [meanvis_after(m_idx),meanaud_after(m_idx)]  = get_data('subject',subjects{m_idx},'expDate',after_string); 
end

%%
figure; 
plot(meanaud_before,meanaud_after,'o',color='red'); 
%legend('aud')
hold on; 
plot(meanvis_before,meanvis_after,'o',color='blue'); 
hold on;
plot([0 1],[0 1],color='black')
hold on;

plot()
%legend(['vis'])

xlim([0 1])
ylim([0 1])
%%

function [meanvis,meanaud] = get_data(varargin)
%% Generate box plots for the behaviour of a mouse/mice
%% Input validation and default assingment
varargin = ['sepPlots', {nan}, varargin];
varargin = ['expDef', {'t'}, varargin];
varargin = ['plotType', {'res'}, varargin];
varargin = ['noPlot', {0}, varargin];
extracted = plt.behaviour.getTrainingData(varargin{:});
for i = find(extracted.validSubjects)'
    if ~isempty(extracted.data{i})
        tDat = extracted.data{i};
        keepIdx = tDat.is_validTrial & ~isnan(tDat.timeline_choiceMoveDir);
        tDat = filterStructRows(tDat, keepIdx);
        if sum(tDat.is_validTrial)>100
            extracted.meanvis_highC{i} = mean(sign(tDat.response_feedback((tDat.is_visualTrial) & (tDat.stim_visContrast>0.2)& (tDat.stim_visContrast<0.3))+1)); 
            extracted.meanaud{i} = mean(sign(tDat.response_feedback((tDat.is_auditoryTrial))+1)); 
        else
            extracted.meanvis_highC{i} = nan; 
            extracted.meanaud{i} = nan; 
        end
    end    
end
meanvis = nanmean(cell2mat(extracted.meanvis_highC));
meanaud = nanmean(cell2mat(extracted.meanaud));
end