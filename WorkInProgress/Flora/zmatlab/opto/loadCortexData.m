
function [siteOrd,alldat] = loadCortexData(which,reverse_hemisphere)


if strcmp('uni',which)
    figPanel = 'a'; 
elseif strcmp('bi',which)
    figPanel = 'u'; 
end 

numRef = double(upper(figPanel)-64);
%
if numRef < 17
    blkDat = spatialAnalysis('all', 'uniscan', 0, 1,'raw');
    laserRef = 1;
    siteOrd = {'Frontal'; 'Vis'; 'Lateral'};
    galvoIdx = {[0.6 2; 1.8, 2; 0.6, 3];[1.8 -4; 3,-4; 3,-3];[4.2,-2; 4.2,-3; 4.2,-4]};
else
    blkDat = spatialAnalysis('all', 'biscan', 0, 1,'raw');
    siteOrd = {'Frontal'; 'Vis'; 'Parietal'};
    laserRef = 2;
    galvoIdx = {[0.5,2;1.5, 2; 0.5, 3];[1.5 -4; 2.5,-4; 2.5,-3];[1.5,-2; 2.5,-2; 3.5,-2]};
end

%pre-assign performance and reaction structures with nans
nMice = length(blkDat.blks);
for mouse = 1:nMice
    iBlk = blkDat.blks(mouse);
    iBlk = prc.filtBlock(iBlk, iBlk.tri.inactivation.galvoPosition(:,2)~=4.5);
    iBlk = prc.filtBlock(iBlk, iBlk.tri.trialType.repeatNum==1 & iBlk.tri.trialType.validTrial);
    iBlk = prc.filtBlock(iBlk, ~isnan(iBlk.tri.outcome.responseCalc));

    if numRef > 16
        %Changes specific to the bilateral inactivation data
        iBlk = prc.filtBlock(iBlk, iBlk.tri.stim.visContrast < 0.07); %
        %this might have been there because on the opto only certain trial
        %types exist for vis (very limited chance to asses sensitivity)
        iBlk.tri.stim.audDiff(isinf(iBlk.tri.stim.audDiff)) = 0;

%         rtLimit = 1.5;
%         iBlk.tri.outcome.responseCalc(iBlk.tri.outcome.reactionTime>rtLimit) = nan;
%         iBlk.tri.outcome.responseRecorded(iBlk.tri.outcome.reactionTime>rtLimit) = 0;
%         iBlk.tri.outcome.reactionTime(iBlk.tri.outcome.reactionTime>rtLimit) = nan;
% 
%         rIdx = iBlk.tri.stim.visDiff>0 | (iBlk.tri.stim.visDiff==0 & iBlk.tri.stim.audDiff>0);
%         iBlk.tri.outcome.responseCalc(rIdx) = (iBlk.tri.outcome.responseCalc(rIdx)*-1+3).*(iBlk.tri.outcome.responseCalc(rIdx)>0);
%         iBlk.tri.stim.audInitialAzimuth(rIdx) = iBlk.tri.stim.audInitialAzimuth(rIdx)*-1;
%         iBlk.tri.stim.visInitialAzimuth(rIdx) = iBlk.tri.stim.visInitialAzimuth(rIdx)*-1;
%         iBlk.tri.stim.visInitialAzimuth(isinf(iBlk.tri.stim.visInitialAzimuth)) = inf;
%         iBlk.tri.stim.conditionLabel(rIdx) = iBlk.tri.stim.conditionLabel(rIdx)*-1;
    else
        iBlk = prc.filtBlock(iBlk, ~ismember(abs(iBlk.tri.inactivation.galvoPosition(:,1)),[0.5; 2; 3.5; 5]) | iBlk.tri.inactivation.laserType==0);
    end

    gPosAbs = [abs(iBlk.tri.inactivation.galvoPosition(:,1)) iBlk.tri.inactivation.galvoPosition(:,2)];

    for site = 1:length(galvoIdx)
        gIdx = ismember(gPosAbs,galvoIdx{site}, 'rows');

        fBlk = prc.filtBlock(iBlk, (iBlk.tri.inactivation.laserType==0 | gIdx));

        normBlk = prc.filtBlock(fBlk, fBlk.tri.inactivation.laserType==0);
        lasBlk = prc.filtBlock(fBlk, fBlk.tri.inactivation.laserType==laserRef);

        % need visdiff, audDiff,response_direction,response_feedback

        ctrlBlock = formatToPinkRigs(normBlk); optoBlock= formatToPinkRigs(lasBlk); 
        ctrlBlock.subjectID_ = ctrlBlock.subjectID*mouse; 
        optoBlock.subjectID_ = optoBlock.subjectID*mouse; 
        controlBlocks{mouse,site} = ctrlBlock; 
        optoBlocks{mouse,site} = optoBlock; 
    end
end


% concatenate control block and opto block into one giant structure across
% mice with adding the mouseID 
allEv = [controlBlocks;optoBlocks]; 

if reverse_hemisphere
   allEv = cellfun(@flip_stimchoice,allEv); 
end 

for regionIdx = 1:size(allEv,2)
    alldat{regionIdx} = concatenateEvents({allEv{:,regionIdx}});
end 

% test whether the fitting works

% f=figure; 
% f.Position = [10,10,400,400];
% plotParams.plottype = 'sig'; 
% 
% plotParams.LineStyle = '--';
% plotParams.DotStyle = 'x';
% plotParams.MarkerEdgeColor = 'k';
% plotParams.MarkerSize = 18; 
% plotParams.LineWidth = 3; 
% plotParams.addFake=0; 
% 
% 
% a = plts.behaviour.GLMmulti(optoBlocks{2,1}, 'simpLogSplitVSplitA');
% a.fit();
% plot_optofit(a,plotParams,1)
end 


function newBlk = formatToPinkRigs(Blk)
        newBlk.stim_visDiff = Blk.tri.stim.visDiff;
        newBlk.stim_audDiff = Blk.tri.stim.audDiff;
        newBlk.response_direction = Blk.tri.outcome.responseCalc; 
        newBlk.response_feedback = Blk.tri.outcome.feedbackGiven;
        newBlk.is_coherentTrial = Blk.tri.trialType.coherent; 
        newBlk.is_conflictTrial = Blk.tri.trialType.conflict;
        newBlk.subjectID = Blk.tri.subjectRef; 
        newBlk.sessionID = Blk.tri.expRef; 
        newBlk.is_laserTrial = logical(Blk.tri.inactivation.laserType);
        newBlk.hemisphere = sign(Blk.tri.inactivation.galvoPosition(:,1)); 
end  

function newBlk = flip_stimchoice(Blk)
         newBlk = Blk;
         newBlk.stim_visDiff = Blk.stim_visDiff.*Blk.hemisphere; 
         newBlk.stim_audDiff = Blk.stim_audDiff.*Blk.hemisphere; 
         newBlk.response_direction = (Blk.response_direction-1.5).*Blk.hemisphere+1.5;
         newBlk = {newBlk};
end 
