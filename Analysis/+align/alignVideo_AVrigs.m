function [tls, blocks, expNums, eTimeline2keep, eBlock2keep, tags, alignDir] = alignEphys_AVrigs(subject, date, P)
    
    subjectsFolder = getRootDir(subject, date);
    alignDir = fullfile(subjectsFolder, 'alignments');
    if ~exist(alignDir,'dir')
        mkdir(alignDir)
    end
    
    [tags, hasEphys] = getEphysTags(subject, date);
    
    % focus on audioVis
    tags = tags(cellfun(@(x) ~isempty(x), strfind(tags,[P.exp '_g'])));
    
    % determine what exp nums exist
    [expNums, blocks, hasBlock, pars, isMpep, tls, hasTimeline] = ...
        whichExpNums(subject, date); % used to be dat.whichExpNums?
    
    useFlipper = true; % no idea why empty
    
    %% align times (timeline to ephys)
    
    %%% not that there should be only 1 exp. Not sure if it will work with
    %%% several of them.
    
    % for any ephys, load the sync data
    if hasEphys
        for t = 1:length(tags)
            if isempty(tags{t})
                [~, pdFlips, allET] = loadSyncChronic(subject, date);
            else
                [~, pdFlips, allET] = loadSyncChronic(subject, date, tags{t});
            end
            if useFlipper
                ephysFlips{t} = allET; % problem with big files, had to bypass spikeGLXdigitalParse
                %             ephysFlips{t} = allET{7}{1};
            else
                ephysFlips{t} = pdFlips;
            end
        end
    end
        
    % detect sync events from timelines
    tlFlips = {};
    for e = 1:length(expNums)
        if hasTimeline(e)
            Timeline = tls{e};
            tt = Timeline.rawDAQTimestamps;
            if useFlipper
                evTrace = Timeline.rawDAQData(:, strcmp({Timeline.hw.inputs.name}, 'flipper'));
                evT = schmittTimes(tt, evTrace, [3 4]); % all flips, both up and down
            else
                evTrace = Timeline.rawDAQData(:, strcmp({Timeline.hw.inputs.name}, 'photoDiode'));
                evT = schmittTimes(tt, evTrace, [3 4]); % all flips, both up and down
                evT = evT([true; diff(evT)>0.2]);
            end
            tlFlips{e} = evT;
        end
    end
    
    % match up ephys and timeline events:
    % algorithm here is to go through each timeline available, figure out
    % whether the events in timeline align with any of those in the ephys. If
    % so, we have a conversion of events in that timeline into ephys
    %
    % Only align to the first ephys recording, since the other ones are aligned
    % to that
    eTimeline2keep = [];
    if hasEphys
        ef = ephysFlips{1};
        if useFlipper && ef(1)<0.001
            % this happens when the flipper was in the high state to begin with
            % - a turning on event is registered at the first sample. But here
            % we need to drop it.
            ef = ef(2:end);
        end
        for e = 1:length(expNums)
            if hasTimeline(e)
                fprintf('trying to correct timeline %d to ephys\n', expNums(e));
                %Timeline = tl{e};
                tlT = tlFlips{e};
                
                success=false;
                if length(tlT)==length(ef)
                    % easy case: the two are exactly coextensive
                    [~,b] = makeCorrection(ef, tlT, true);
                    success = true;
                elseif length(tlT)<length(ef) && ~isempty(tlT)
                    [~,b,success] = findCorrection(ef, tlT, false);
                elseif length(tlT)>length(ef) && ~isempty(tlT)
                    [~,a,success] = findCorrection(tlT, ef, false);
                    if ~isempty(a)
                        b = [1/a(1); -a(2)/a(1)];
                    end
                end
                if success
                    %                 writeNPY(b, fullfile(alignDir, ...
                    %                     sprintf('correct_timeline_%d_to_ephys_%s.npy', ...
                    %                     e, tags{1})));
                    bTL2ephys(:,e) = b;
                    writeNPY(b, fullfile(alignDir, ...
                        sprintf('correct_timeline_%d_to_ephys_%s.npy', ...
                        expNums(e), tags{1})));
                    fprintf('success\n');
                    eTimeline2keep = [eTimeline2keep e];
                else
                    fprintf('could not correct timeline to ephys\n');
                end
            end
        end
    end
    
    %% Load each Timeline/Block pair
    
    eBlock2keep = [];
    
    for e = 1:length(expNums)
        if sum(eTimeline2keep == e)>0 && hasTimeline(e) && hasBlock(e) && contains(blocks{e}.expDef,P.expdef)
            
            eTL = e; % should be the same
            if useFlipper
                % didn't get photodiode flips above, so get them now
                Timeline = tls{eTL};
                tt = Timeline.rawDAQTimestamps;
                evTrace = Timeline.rawDAQData(:, strcmp({Timeline.hw.inputs.name}, 'photoDiode'));
                pdT = schmittTimes(tt, evTrace, [8 9]);
                if isempty(pdT)
                    pdT = schmittTimes(tt, evTrace, [2.5 3]);
                end
            else
                pdT = tlFlips{eTL};
            end
%             pdT = pdT(2:end);
            
            fprintf('trying to correct block %d to timeline %d\n', expNums(e), expNums(eTL));
            
            block = blocks{e};
            sw = block.stimWindowUpdateTimes;
            sw = sw(2:end); % sometimes need this? Why? how did sw
            % get an extra event at the beginning?
            
            success = false;
            if length(sw)<length(pdT) && length(sw)>1
                [~,b,success,actualTimes] = findCorrection(pdT, sw, false);
                if ~success
                    % might be because photodiode acted strangely
                    % find unexpected photodiode events
                    unexp_idx = find(diff(pdT) < min(diff(sw)));
                    % remove them
                    pdT(unexp_idx(2):unexp_idx(2)+1) = [];
                    [~,b,success,actualTimes] = findCorrection(pdT, sw, false);
                end
            end
            if length(sw)==length(pdT)
                % easy case: the two are exactly coextensive
                [~,b] = makeCorrection(pdT, sw, true);
                success = true;
            end
            if success
                %                     writeNPY(b, fullfile(alignDir, ...
                %                         sprintf('correct_block_%d_to_timeline_%d.npy', ...
                %                         e, eTL)));
                %                     writeNPY(actualTimes, fullfile(alignDir, ...
                %                         sprintf('block_%d_sw_in_timeline_%d.npy', ...
                %                         e, eTL)));
                writeNPY(b, fullfile(alignDir, ...
                    sprintf('correct_block_%d_to_timeline_%d.npy', ...
                    expNums(e), expNums(eTL))));
                
                fprintf('  success\n');
                eBlock2keep = [eBlock2keep e];
            else
                fprintf('  could not correct block %d to timeline %d\n', expNums(e), expNums(eTL));
            end
        elseif isMpep(e)
            % take it from Anna's script if needed
        end
    end
    
    TL = tls(eBlock2keep);
    BLOCKS = blocks(eBlock2keep);
end
