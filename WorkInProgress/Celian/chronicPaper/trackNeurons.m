UMFiles = {"\\znas.cortexlab.net\Lab\Share\UNITMATCHTABLES_ENNY_CELIAN_JULIE\FullAnimal_new\AV009\Probe1\IMRO_10\UnitMatch\UnitMatch.mat"};
load(UMFiles{1})
probe = csv.checkProbeUse('AV009');

convertedDays = cell2mat(cellfun(@(x) datenum(x(37:46)), UMparam.KSDir, 'uni', 0)) - datenum(probe.implantDate{1});

%%
summaryFunctionalPlots(UMFiles,'Corr',1)

%%
[unitPresence, unitProbaMatch, days] = summaryMatchingPlots(UMFiles,1,0);

%% Compare to cemented probes?

%% Find tracked population

subSelec = find(any(unitPresence{midx}(1,:),1));
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
rec = [1 5 10 15 20 25];

figure;
imagesc(unitPresence{1}(rec,unitPresence{1}(1,:) == 1)')

popUIDs = UIDuni(all(unitPresence{1}(rec,unitPresence{1}(1,:) == 1)));

%% Show waveforms and ISIs

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


%%

[u,s,v] = svd(nanmean(wav,3)');
[~,unitSortIdx] = sort(u(:,1));
col = rand(numel(popUIDs),3); %colorcube(numel(popUIDs)*2); col(numel(popUIDs)+1:end,:) = [];

cell2plt = [1 23 24];

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
for cc = 1:numel(cell2plt)
    text(xloc(unitSortIdx(cell2plt(cc)),rr), yloc(unitSortIdx(cell2plt(cc)),rr),num2str(cc))
end
linkaxes(s,'xy')

figure('Position', [585   638   655   340]);
% colDays = [linspace(160,24,numel(rec)); linspace(238,128,numel(rec)); linspace(157,100,numel(rec))]'/256;
colDays = [linspace(0.7,0,numel(rec)); linspace(0.7,0,numel(rec)); linspace(0.7,0,numel(rec))]';
for cc = 1:numel(cell2plt)
    for rr = 1:numel(rec)
        subplot(2, numel(cell2plt), cc) % Waveforms
        hold all
        w = wav(:,unitSortIdx(cell2plt(cc)),rr);
        plot(w - nanmean(w(1:20))-0.3*max(wav(:,unitSortIdx(cell2plt(cc)),1))*rr,'color',colDays(rr,:), 'LineWidth', 2.0)
        if rr == numel(rec); makepretty; offsetAxes; end
        if cc == 1 && rr == 1; ylabel('Waveform'); end

        subplot(2, numel(cell2plt), numel(cell2plt)+cc) % ISI
        hold all
        stairs(ISIbins(1:end-1)*1000, isi(:,unitSortIdx(cell2plt(cc)),rr)-0.01*rr,'color',colDays(rr,:), 'LineWidth', 2.0);
        xticks([5 50 500])
        yticks([0 0.07])
        xlabel('Time (ms)')
        if cc == 1 && rr == 1; ylabel('Firing rate (sp/s)'); end
        if rr == numel(rec); legend({num2str(convertedDays')}); set(gca,'XScale','log'); makepretty; end
    end
end