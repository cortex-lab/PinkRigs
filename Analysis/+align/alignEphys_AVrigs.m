function b = alignEphys_AVrigs(subject, date, P)
    
    subjectsFolder = getRootDir(subject, date);
    alignDir = fullfile(subjectsFolder, 'alignments');
    
    [tags, hasEphys] = getEphysTags(subject, date);
    
    % focus on audioVis
    tags = tags(cellfun(@(x) ~isempty(x), strfind(tags,P.exp)));
    
    % determine what exp nums exist
    [expNums, blocks, hasBlock, pars, isMpep, tl, hasTimeline] = ...
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
            Timeline = tl{e};
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
                    writeNPY(b, fullfile(alignDir, ...
                        sprintf('correct_timeline_%d_to_ephys_%s.npy', ...
                        expNums(e), tags{1})));
                    fprintf('success\n');
                    eTimeline2keep = e;
                else
                    fprintf('could not correct timeline to ephys\n');
                end
            end
        end
    end
    
    TLexp = expNums(eTimeline2keep);
    Timeline = tl{eTimeline2keep};
    
end