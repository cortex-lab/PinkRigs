
% goal of this script is to plot the bias parameter of the ephys mice
% across time  - now I will plot it across trials
 
clc; clear all; 

subjects = {['AV034']}%,['AV020'],['AV025'],['AV030'],['AV034']};
sepSessions = 1; 

for mys=1:numel(subjects)
    clear extracted events;
    currSubject = subjects{mys};%;'AV030';
    extracted = plts.behaviour.getTrainingData('subject', {currSubject}, 'expDate', '2022-12-16', 'sepPlots', 1); 
    %
    % add some information to each, e.g. sessionID, nogoblock
    %  throw away sessions with less that x trials 
    
    unique_subjects = unique(extracted.subject);
    subject_indices=1:numel(unique_subjects);
    dates = extracted.blkDates; 

    for i=1:numel(extracted.data)
        events = extracted.data{i, 1}; 
        nTrials = numel(events.is_blankTrial);
        nKept = sum(events.is_validTrial & events.response_direction & abs(events.stim_audAzimuth)~=30); 
        extracted.nTrials{i,1} = nTrials;
        extracted.nKept{i,1} = nKept;
        extracted.data{i, 1}.sessionID = ones(nTrials,1)*i;
        extracted.data{i, 1}.subjectID  = ones(nTrials,1)*subject_indices(strcmp(unique_subjects,extracted.subject{i})); 
        

        % get nogoBlock changepoints
        % 
        blockID = ones(nTrials,1); 
        n_nogos = 5; n_gos = 5; % basically we find that pattern of x nogos followed by x gos
        convkernel = [ones(n_nogos,1);zeros(n_gos,1)];        
        isnogo = events.response_direction==0;        
        nogo_goswitch_idx = strfind(isnogo', convkernel');
        go_nogo_switch = strfind(isnogo', flipud(convkernel)');
        nogo_goswitch_idx = sort([nogo_goswitch_idx,go_nogo_switch]);
        figure; plot(~isnogo); ylim([-.2,1.2])
        if numel(nogo_goswitch_idx)>0
            for s=1:numel(nogo_goswitch_idx)
                curr_switch_idx = nogo_goswitch_idx(sepSessions);
                vline(curr_switch_idx); 
                blockID(curr_switch_idx:end) = blockID(curr_switch_idx:end)+1; 
            end
        end 
        extracted.data{i, 1}.blockID = blockID;
    end 
    %
    % filter sessions that do not pass the nTrial test
    minValid = 100; 
    extracted.validSubjects = num2cell(extracted.validSubjects);
    extracted  = filterDataStruct(extracted,([extracted.nKept{:}]>minValid));
    
    % after filtering we add the blockIDs 
    % nParts = 100; % number of trials to partition the session to
    % for i=1:numel(extracted.data)
    %     % prepare 
    % 
    % 
    % 
    %     nTrials = extracted.nTrials{i,1};
    %     nBlocks = floor(nTrials/nParts);
    %     blockIDs = 1:nBlocks; 
    %     currIDs = repelem(blockIDs,nParts); 
    %     % put the remainder in the last block
    %     currIDs  = [currIDs,repelem(blockIDs(end),nTrials-numel(currIDs))]; 
    %     extracted.data{i, 1}.blockID = currIDs';
    % 
    % end 
    
    events = concatenateEvents(extracted.data); 
    % throw away trials, that are A) not valid, nogo, or the aud azimuth is 30 
    events = filterStructRows(events, (events.is_validTrial & ...
        events.response_direction & abs(events.stim_audAzimuth)~=30));  % could do this potentailly earlier...
    
    % repartition the struct based on the unique charachterisitic critera
    
    
    paramSet = unique([events.sessionID,events.blockID], 'rows');
    for i=1:size(paramSet,1)
       event_subset =filterStructRows(events,events.sessionID==paramSet(i,1) & events.blockID==paramSet(i,2));
       events_.data{i,1} = event_subset;
    
    end 
    events=events_; clear events_;
    
    % options: per session, per n trial, % x per nogo blocks.
    %fit to each set
    %
    for s=1:numel(events.data)
        currBlock = events.data{s};
        glm = plts.behaviour.GLMmulti(currBlock, 'simpLogSplitVSplitA');
        glm.fit;
        bias(s) = glm.prmFits(1); 
        sessdate(s) = datetime(dates{currBlock.sessionID(1)}); 
    end 
    % %% 
    %
    
    figure; 

    plot(sessdate,bias,'o-')
    hline(0,'k--')
    xlabel('sessionID'); 
    ylabel('bias');

    title(currSubject); 
    ylim([-3,3])
end

%%

% the above atm does not work because you need to exclude valid trials. %
% it will work eventually when I write code 
%%
% subjects = {['AV034']};%,['AV020'],['AV025'],['AV030'],['AV034']};
% for s=1:numel(subjects)
%     currSubject =  subjects{s}; 
%     expList = csv.queryExp(subject=currSubject,expDate='postImplant',expDef='t',sepPlots=1); 
%     glm = plts.behaviour.glmFit(expList); 
%     
%     keepExps = ~cellfun(@isempty,glm);
%     expList = expList(keepExps,:); glm = glm(keepExps);
%     %
%     
%     biases = cellfun(@(x) x.prmFits(1),glm); 
%     expDurs = str2double([expList.expDuration(:)]); 
%     expDates = [expList.expDate(:)]; 
%     %
%     toKeep = expDurs>1500;  
%     f = figure;
%     dates = datetime(expDates(toKeep)); 
%     plot(dates,biases(toKeep),'o-');
%     title(currSubject);
%     ylabel('bias'); 
%     hline(0,'k--')
%     
%     dateticks = datenum(dates); 
%     ticks= datetime(dateticks(1):dateticks(end),'ConvertFrom','datenum');
%     set(gca,'XTick',ticks);
%     datetick('x','mm-dd','keepticks'); 
%     set(gca,'XTickLabelRotation',60)
% end
% filter for expDuration