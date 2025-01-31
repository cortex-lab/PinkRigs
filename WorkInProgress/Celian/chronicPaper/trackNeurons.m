% UMFiles = {"\\znas.cortexlab.net\Lab\Share\UNITMATCHTABLES_ENNY_CELIAN_JULIE\FullAnimal_KSChanMap\AV009\Probe1\IMRO_10\UnitMatch\UnitMatch.mat"};
% UMFiles = {"D:\MatchingUnits\Data\UnitMatch\UnitMatch.mat"};
UMFiles = {"\\znas.cortexlab.net\Lab\Share\Celian\UnitMatch\MatchTables_paper_oldAssignUIDAlgo\AV009\19011118541\3\UnitMatch\UnitMatch.mat"};
% UMFiles = {"\\znas.cortexlab.net\Lab\Share\Celian\UnitMatch\MatchTables_paper\AV009\19011118541\3\UnitMatch\UnitMatch.mat"};
load(UMFiles{1})
probe = csv.checkProbeUse('AV009');

convertedDays = cell2mat(cellfun(@(x) datenum(x(37:46)), UMparam.KSDir, 'uni', 0)) - datenum(probe.implantDate{1});

%%
% summaryFunctionalPlots(UMFiles,'Corr',1)

%%
[unitPresence, unitProbaMatch, days] = summaryMatchingPlots(UMFiles,'UID1',1,0);

%% Compare to cemented probes?

%% Find tracked population

midx = 1;
subSelec = find(any(unitPresence{midx}(1:3,:),1));
figure;
[numRec,sortIdx] = sort(sum(unitPresence{midx}(:,subSelec)), 'descend');
subplot(121)
imagesc(unitPresence{midx}(:,subSelec(sortIdx))')
c = colormap('gray'); c = flipud(c); colormap(c)
caxis([0 1])
ylabel('Unit')
xlabel('Days')
xticks(1:numel(convertedDays));
xticklabels(num2str(convertedDays'))
subplot(122)
plot(numRec, 1:numel(numRec),'k');
set(gca,'YDir','reverse')
axis tight

UDtoUse = 'UID1';
UIDuni = unique([MatchTable.(UDtoUse)]);
% rec = [1 5 10 15 20 25];
 rec = [2 5 10 15 19 25 27];
% rec = [1 5 11 15 21 25 27];

figure;
imagesc(unitPresence{1}(rec,unitPresence{1}(1,:) == 1)')

% popUIDs = UIDuni(all(unitPresence{1}(rec,unitPresence{1}(1,:) == 1)));
popUIDs = UIDuni(all(unitPresence{1}(rec,:)));

%% Get waveforms and ISIs

ISIbins = [0 5*10.^(-4:0.1:0)];
probeType = 'NaN';

unitIdx = 1;
wav = nan(82,numel(popUIDs),numel(rec));
xloc = nan(numel(popUIDs),numel(rec));
yloc = nan(numel(popUIDs),numel(rec));
isi = nan(numel(ISIbins)-1,numel(popUIDs),numel(rec));
for rr = 1:numel(rec)
    fprintf('rec %d.\n', rr)
    KSDir = UMparam.KSDir{rec(rr)};

    load(fullfile(KSDir, 'PreparedData.mat'), 'sp');

    % Get scaling factor
    metaFile = regexprep(sp.dat_path(5:end-1), 'ap.cbin', 'ap.meta');
    [scalingFactor, ~, ~] = bc_readSpikeGLXMetaFile(metaFile, probeType);
    
    for uidx = 1:numel(popUIDs)
        unitID = MatchTable(find(MatchTable.(UDtoUse) == popUIDs(uidx) & MatchTable.RecSes1 == rec(rr),1),:).ID1;

        % Load waveform
        tmppath = dir(fullfile(KSDir,'**','RawWaveforms*'));
        Path4UnitNPY = fullfile(tmppath.folder,tmppath.name,['Unit' num2str(unitID) '_RawSpikes.npy']);
        spikeMap = readNPY(Path4UnitNPY);
        spikeMap = nanmean(spikeMap,3)*scalingFactor;
        [m,idx] = max(abs(spikeMap));
        [mt,idx1] = max(m);
        wav(:,uidx,rr) = spikeMap(:,idx1);

        % Get location
        xloc(uidx,rr) = sp.templateXpos(sp.cids == unitID);
        yloc(uidx,rr) = sp.templateDepths(sp.cids == unitID);
        
        % Compute ISI
        idx = sp.spikeTemplates == unitID;
        isi(:,uidx,rr) = histcounts(diff(double(sp.st(idx))),ISIbins, 'Normalization','probability');
    end
end


%% Show

[u,s,v] = svd(nanmean(wav,3)');
[~,unitSortIdx] = sort(u(:,1));
col = rand(numel(popUIDs),3); %colorcube(numel(popUIDs)*2); col(numel(popUIDs)+1:end,:) = [];

figure;
for rr = 1:numel(rec)
    s(rr) = subplot(3, numel(rec), rr); % location
    scatter(xloc(:,rr), yloc(:,rr), 20, col, 'filled', 'AlphaData', 0.5)
    if rr == 1; ylabel('depth'); end
    title(UMparam.KSDir{rec(rr)}(31:46))

    subplot(3, numel(rec), numel(rec)+rr) % Waveforms
    imagesc(zscore(wav(:,unitSortIdx,rr))')
%     clim([-700 100])
    if rr == 1; ylabel('Waveform'); end

    subplot(3, numel(rec), 2*numel(rec)+rr) % ISIs
    imagesc(isi(:,unitSortIdx,rr)')
    clim([0 0.07])
    if rr == 1; ylabel('ISI'); end
end
linkaxes(s,'xy')
if exist('cell2plt','var')
    subplot(3, numel(rec), 1);
    for cc = 1:numel(cell2plt)
        text(xloc(unitSortIdx(cell2plt(cc)),rr), yloc(unitSortIdx(cell2plt(cc)),rr),num2str(cc))
    end
end

%% Single cells

% cell2plt = [1 23 24];
% cell2plt = [1 4 8 11];
% cell2plt = [9 15 16];
% cell2plt = [1 2];
% cell2plt = [1 12 28];
% cell2plt = [15 18 19];   
cell2plt = [4 17 23 25];   

figure('Position', [585   638   655   340]);
% colDays = [linspace(160,24,numel(rec)); linspace(238,128,numel(rec)); linspace(157,100,numel(rec))]'/256;
colDays = [linspace(0.7,0,numel(rec)); linspace(0.7,0,numel(rec)); linspace(0.7,0,numel(rec))]';
clear s
for cc = 1:numel(cell2plt)
    for rr = 1:numel(rec)
        s1(cc) = subplot(2, numel(cell2plt), cc); % Waveforms
        hold all
        w = wav(:,unitSortIdx(cell2plt(cc)),rr);
        plot(w - nanmean(w(1:20))-20*rr,'color',colDays(rr,:), 'LineWidth', 2.0)
        if rr == numel(rec); makepretty; offsetAxes; end
        if cc == 1 && rr == 1; ylabel('Waveform'); end

        s2(cc) = subplot(2, numel(cell2plt), numel(cell2plt)+cc); % ISI
        hold all
        stairs(ISIbins(1:end-1)*1000, isi(:,unitSortIdx(cell2plt(cc)),rr)-0.01*rr,'color',colDays(rr,:), 'LineWidth', 2.0);
        xticks([5 50 500])
        yticks([0 0.07])
        xlabel('Time (ms)')
        if cc == 1 && rr == 1; ylabel('Firing rate (sp/s)'); end
        if rr == numel(rec); legend({num2str(convertedDays(rec)')}); set(gca,'XScale','log'); makepretty; end
    end
end
linkaxes(s1, 'xy')
linkaxes(s2, 'xy')

%% Summary

% subjectList = {'AV008','AV009','AV015','AV021','AV049','CB015','CB016','CB017','CB018','CB020'};
% % subjectList = {'AL031', 'AL032', 'AL036'};
% 
% d = [];
% for ss = 1:numel(subjectList)
%     dtmp = dir(fullfile('\\znas.cortexlab.net\Lab\Share\Celian\UnitMatch\MatchTables_paper_oldAssignUIDAlgo\,subjectList{ss},'**','UnitMatch.mat'))
% %     dtmp = dir(fullfile('\\znas.cortexlab.net\Lab\Share\Celian\UnitMatch\MatchTables\',subjectList{ss},'**','UnitMatch.mat'));
%     % dtmp = dir(fullfile('\\znas.cortexlab.net\Lab\Share\UNITMATCHTABLES_ENNY_CELIAN_JULIE\FullAnimal_KSChanMap',subjectList{ss},'**','UnitMatch.mat'));
%     d = cat(1, d, dtmp);
% end

d = dir(fullfile('\\znas.cortexlab.net\Lab\Share\Celian\UnitMatch\MatchTables_paper_oldAssignUIDAlgo\','**','UnitMatch.mat'));

UMFiles = cell(1,numel(d));
clear subj
for midx = 1:numel(d)
    UMFiles{midx} = fullfile(d(midx).folder, d(midx).name);
    [~,subj{midx}] = fileparts(fileparts(fileparts(fileparts(d(midx).folder))));
end
[~, ~, groupVector] = unique(subj);

% Plot summary across mice
% res = summaryFunctionalPlots(UMFiles, 'Corr', groupVector);
[unitPresence, unitProbaMatch, days, EPosAndNeg, DataSetInfo, pSinceAppearance, popCorr_Uni, popAUC_Uni] = summaryMatchingPlots(UMFiles,'UID1',groupVector,1);
% trackWithFunctionalMetrics(UMFiles);

%% Make plot grouping by animal

groups = unique(groupVector);
groupColor = distinguishable_colors(max(groups)+1);

deltaDaysBinsOri = 2.^(-1:8);
deltaDaysBins = [-deltaDaysBinsOri(end:-1:1)-0.1,deltaDaysBinsOri(1:end)];
deltaBinsVec =[-deltaDaysBinsOri(end:-1:1),deltaDaysBinsOri];
yTickLabels = arrayfun(@(X) num2str(round(X*10)/10),deltaBinsVec(1:end-1),'Uni',0);

fnames = fieldnames(popCorr_Uni);

nNeur = cell2mat(cellfun(@(x) size(x,2), unitPresence, 'uni',0));

group2plot = unique(groupVector(find(contains(subj,'AV009'))));
colGroup = [0.4157    0.2392    0.6039];
figure;

% Matching proba
subplot(2,1+numel(fnames),1)
hold on
clear pSinceAppearance_perGroup
for gg = 1:length(groups)
    gIdx = groupVector == groups(gg);
    pSinceAppearance_perGroup(:,gg) = nansum(pSinceAppearance(:,gIdx).*nNeur(gIdx),2)./(~isnan(pSinceAppearance(:,gIdx))*nNeur(gIdx)');
    plot(pSinceAppearance_perGroup(:,gg),'color',[.5 .5 .5])
end
% plot(pSinceAppearance(:,find(contains(UMFiles,'AV009\19011118541\3'))),'color',colGroup)
plot(pSinceAppearance_perGroup(:,group2plot),'color',colGroup)
xlabel('delta Days')
set(gca,'XTick',1:numel(deltaDaysBins)-1,'XTickLabel',yTickLabels)
ylabel('P(track)')
nonnanNr = sum(~isnan(pSinceAppearance_perGroup),2);
h = errorbar(1:size(pSinceAppearance_perGroup,1),nanmean(pSinceAppearance_perGroup,2),nanstd(pSinceAppearance_perGroup,[],2)./sqrt(nonnanNr-1),'linestyle','-','color','k');
h.LineWidth = 2;
offsetAxes

% Number of datasets
subplot(2,1+numel(fnames),2+numel(fnames))
hold on
scatter(1:size(pSinceAppearance,1),nonnanNr,20,'k','filled')
set(gca,'XTick',1:numel(deltaDaysBins)-1,'XTickLabel',yTickLabels)
ylabel('# animals')
offsetAxes

for ff = 1:numel(fnames)
    % Functional fingerprint correlation
    subplot(2,1+numel(fnames),1+ff)
    hold on
    clear popCorr_Uni_perGroup
    for gg = 1:length(groups)
        popCorr_Uni_perGroup(:,gg) = nanmean(popCorr_Uni.(fnames{ff})(:,groupVector == groups(gg)),2);
        plot(popCorr_Uni_perGroup(:,gg),'color',[.5 .5 .5])
    end
    plot(popCorr_Uni_perGroup(:,group2plot),'color',colGroup)
    h = errorbar(1:size(popCorr_Uni_perGroup,1),nanmean(popCorr_Uni_perGroup,2),nanstd(popCorr_Uni_perGroup,[],2)./sqrt(nonnanNr-1),'linestyle','-','color','k');
    h.LineWidth = 2;
    xlabel('delta Days')
    set(gca,'XTick',1:numel(deltaDaysBins)-1,'XTickLabel',yTickLabels)
    ylabel('corr')
    title(fnames{ff})
    ylim([-0.5 1])
    offsetAxes

    % Functional fingerprint AUC
    subplot(2,1+numel(fnames),2+numel(fnames)+ff)
    hold on
    clear popAUC_Uni_perGroup
    for gg = 1:length(groups)
        popAUC_Uni_perGroup(:,gg) = nanmean(popAUC_Uni.(fnames{ff})(:,groupVector == groups(gg)),2);
        plot(nanmean(popAUC_Uni.(fnames{ff})(:,groupVector == groups(gg)),2),'color',[.5 .5 .5])
    end
    plot(popAUC_Uni_perGroup(:,group2plot),'color',colGroup)
    h = errorbar(1:size(popAUC_Uni_perGroup,1),nanmean(popAUC_Uni_perGroup,2),nanstd(popAUC_Uni_perGroup,[],2)./sqrt(nonnanNr-1),'linestyle','-','color','k');
    h.LineWidth = 2;
    xlabel('delta Days')
    set(gca,'XTick',1:numel(deltaDaysBins)-1,'XTickLabel',yTickLabels)
    ylabel('AUC')
    title(fnames{ff})
    ylim([0 1])
    offsetAxes
end