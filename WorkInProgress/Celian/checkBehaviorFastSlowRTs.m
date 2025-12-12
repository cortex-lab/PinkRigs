exp = csv.queryExp(subject='GB012', expDate = 'last120', expDef ='t', checkEvents = 1, sepPlots = 1, noPlot = 1);
behData = plts.behaviour.glmFit(exp, modelString = 'simpLogSplitVSplitA');

%%
clear glmDataFast glmDataSlow blockIsAud dom domFast domSlow RTmean
for bb = 1:numel(behData) 
    currBlock = behData{bb}.dataBlock;
    RTs = currBlock.timeline_choiceMoveOn - currBlock.timeline_audPeriodOn;
    fastIdx = RTs < 0.1;
    RTmean(bb) = nanmean(RTs);

    % Detect which block we're in
    confIdx = currBlock.is_conflictTrial == 1;
    blockIsAud(bb) = all((currBlock.stim_audAzimuth(confIdx) == -60) +1 == currBlock.stim_correctResponse(confIdx));

%     % fast ones
%     fastBlock = filterStructRows(currBlock,fastIdx);
%     glmDataFast{bb} = plts.behaviour.GLMmulti(fastBlock, 'simpLogSplitVSplitA');
%     glmDataFast{bb}.fit;
% 
%     % slow ones
%     slowBlock = filterStructRows(currBlock,~fastIdx);
%     glmDataSlow{bb} = plts.behaviour.GLMmulti(slowBlock, 'simpLogSplitVSplitA');
%     glmDataSlow{bb}.fit;

    % quick dom
    conf1Idx = currBlock.is_validTrial & currBlock.stim_audAzimuth == -60 & currBlock.stim_visAzimuth == 60 & currBlock.stim_visContrast > 0.15;
    conf2Idx = currBlock.is_validTrial & currBlock.stim_audAzimuth == 60 & currBlock.stim_visAzimuth == -60 & currBlock.stim_visContrast > 0.15;
    dom(bb) = sum(currBlock.response_direction(conf1Idx) == 1)/sum(conf1Idx) - sum(currBlock.response_direction(conf2Idx) == 1)/sum(conf2Idx);
    domFast(bb) = sum(currBlock.response_direction(conf1Idx & fastIdx) == 1)/sum(conf1Idx & fastIdx) - sum(currBlock.response_direction(conf2Idx & fastIdx) == 1)/sum(conf2Idx & fastIdx);
    domSlow(bb) = sum(currBlock.response_direction(conf1Idx & ~fastIdx) == 1)/sum(conf1Idx & ~fastIdx) - sum(currBlock.response_direction(conf2Idx & ~fastIdx) == 1)/sum(conf2Idx & ~fastIdx);
end

% prmFitsAll = cell2mat(cellfun(@(x) x.prmFits, behData, 'uni', 0))';
% prmFitsFast = cell2mat(cellfun(@(x) x.prmFits', glmDataFast, 'uni', 0));
% prmFitsSlow = cell2mat(cellfun(@(x) x.prmFits', glmDataSlow, 'uni', 0));

switches = find(diff(blockIsAud) ~= 0)+1;

%%

% figure;
% subplot(311); hold all
% plot(nanmean(prmFitsAll(contains(behData{1}.prmLabels,'vis'),:),1),'k')
% plot(nanmean(prmFitsAll(contains(behData{1}.prmLabels,'aud'),:),1),'g')
% vline(switches)
% subplot(312); hold all
% plot(nanmean(prmFitsFast(contains(behData{1}.prmLabels,'vis'),:),1),'k')
% plot(nanmean(prmFitsFast(contains(behData{1}.prmLabels,'aud'),:),1),'g')
% vline(switches)
% subplot(313); hold all
% plot(nanmean(prmFitsSlow(contains(behData{1}.prmLabels,'vis'),:),1),'k')
% plot(nanmean(prmFitsSlow(contains(behData{1}.prmLabels,'aud'),:),1),'g')
% vline(switches)

figure;
subplot(211)
hold all
plot(dom,'k')
plot(domFast,'r')
plot(domSlow,'g')
ylabel('Dominance')
vline(switches)

subplot(212)
hold all
plot(RTmean)
vline(switches)
xlabel('sessions')
ylabel('RT (s)')

figure;
win = -2:5;
for ss = 1:numel(switches)
    if rem(ss,2) == 0
        subplot(121)
    else
        subplot(122)
    end
    hold all
    plot(win, dom(min(switches(ss)+(win),numel(domFast))),'k')
    plot(win, domFast(min(switches(ss)+(win),numel(domFast))),'r')
    plot(win, domSlow(min(switches(ss)+(win),numel(domFast))),'g')
    ylim([-1 1])
end
subplot(121)
vline(0)
subplot(122)
vline(0)


% %%
% behDataBoxPlot = plts.behaviour.boxPlots(subject='GB002', expDate ='last60', sepPlots = 1, noPlot = 1);
% perf = cell2mat(cellfun(@(x) x.plotData(end,end)-x.plotData(1,1), behDataBoxPlot, 'uni', 0));
% dom = cell2mat(cellfun(@(x) x.plotData(end,1)-x.plotData(1,end), behDataBoxPlot, 'uni', 0));
% 
% %%
% figure;
% hold all
% plot(perf)
% plot(dom)
% vline(switches)

