function master(subject, expDate, expNum)
    
%%% Should the functions output the times? Or save it and we actually load  
%%% it somewhere else?    
    
%% Align spike times to timeline and save results in experiment folder
%  This function will load the timeline flipper from the experiment and
%  check this against all ephys files recorded on the same date. If it
%  locates a matching section of ephys data, it will extract the
%  corresponding spike data and save it into the experiment folder. This
%  function will not run if there is no ephys/kilosort data


%% Align the block timings to timeline
%  This function will load the timeline and block for that experiment and
%  align one with another using 1) the wheel or 2) the photodiode.

[blockTime, timelineTime] = align.block_AVrigs(subject, expDate, expNum);


%% Align the video frame times to timeline
%  This function will align all cameras' frame times with the experiment's
%  timeline.
%  The resulting times for these alignments will be saved in a structure
%  'vids' that contains all cameras.

% a few parameters (optional)
pVid.recomputeAlign = false; % will recompute the alignment if true
pVid.recomputeInt = false; % will recompute intensity file if true
pVid.nFramesToLoad = 3000; % will start loading the first and 3000 of the movie
pVid.adjustPercExpo = 1; % will adjust the timing of the first frame from its intensity
pVid.plt = 1; % to plot the inter frame interval for sanity checks
pVid.crashMissedFrames = 1; % will crash if any missed frame

% get cameras' names
expPath = getExpPath(subject, expDate, expNum);
vids = dir(fullfile(expPath,'*Cam.mj2')); % there should be 3: side, front, eye

% align each of them
for v = 1:numel(vids)
    [~,vidName,~]=fileparts(vids(v).name);
    try
        [vids(v).frameTimes, vids(v).missedFrames] = align.video_AVrigs(subject, expDate, expNum, vidName, pVid);
    catch me
        % case when it's corrupted
        vids(v).frameTimes = [];
        vids(v).missedFrames = [];
        warning('Corrupted video %s: threw an error (%s)',vidName,me.message)
    end
end

%% Align microphone to timeline
%  This function will take the output of the 192kHz microphone and align it
%  to the low frequency microphone that records directly into the timeline
%  channel--HOW WILL THIS BE SAVED?



