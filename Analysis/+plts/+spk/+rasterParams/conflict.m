function [eventTimes, trialGroups, opt] = conflict(ev)

    opt = struct;
    choiceLabels = (ev.response_direction*2-3).*(ev.response_direction>0);
    choiceNames = {'MoveL'; 'Timeout'; 'MoveR'}; 

    idx = (ev.is_conflictTrial) & (ev.stim_audAzimuth==60);     

    
    eventTimes = {...
        ev.timeline_visPeriodOnOff(idx,1); ...
        ev.timeline_audPeriodOnOff(idx,1);

        };

    opt.eventNames = {...
        'Vis Onset'; ...
        'Aud Onset'; ...
        };

    trialGroups = {...
        [choiceLabels(idx)]; ...
        [choiceLabels(idx)];

        };

    opt.groupNames = {...
        {'Choice'} ...
        {'Choice'} 
        };


    %         opt.groupNames = {...
%             {{'visL'; 'visR'}, choiceNames, trialNames} ...
%             {{'audL'; 'aud0'; 'audR'}, choiceNames, trialNames} ...
%             {choiceNames, trialNames} ...
%             };


    opt.sortClusters = 'sig';
    opt.sortTrials = repmat({ev.timeline_choiceMoveOn(idx)}, length(eventTimes),1);
    opt.trialTickTimes = repmat({ev.timeline_choiceMoveOn(idx)}, length(eventTimes),1);
end


