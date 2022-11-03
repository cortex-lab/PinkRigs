%% ------ Stability plots ------
%% Get data

% [clusterNum, recLocAll, days] = plt.spk.clusterCount(pltIndiv=0);
% save('\\zserver\Lab\Share\Celian\dataForSfn2022_ChronicImplant_stability','clusterNum', 'recLocAll', 'days')

load('\\zserver.cortexlab.net\Lab\Share\Celian\dataForSfn2022_ChronicImplant_stability')

%% Plot it
recInfo = cellfun(@(x) split(x,'__'),recLocAll,'uni',0);
subjectsAll = cellfun(@(x) x{1}, recInfo, 'UniformOutput', false);
probeSNAll = cellfun(@(x) x{2}, recInfo, 'UniformOutput', false);

subjects = unique(subjectsAll);
colAni = lines(numel(subjects));

fullProbeScan = {{'0__0'}, {'1__0'}, {'2__0'}, {'3__0'}, ...
    {'0__2880'}, {'1__2880'}, {'2__2880'}, {'3__2880'}};

slopeMean = nan(numel(subjects),2);
figure
% figure('Position',[680   728   450   250]);
hold all

% Find those that match location
% Do it by subject and probe so that it's easier to do the whole probe
% thing...?
pltIndiv = 0;
recLocSlope = cell(1,1);
b = cell(1,1);
useNum = nan(numel(subjects),2);
for ss = 1:numel(subjects)
    subjectIdx = contains(subjectsAll,subjects(ss));
    probes = unique(probeSNAll(subjectIdx));
    for pp = 1:numel(probes)

        % Check number of uses for this probe
        SN = str2double(probes(pp));
        probeInfo = csv.checkProbeUse(SN);
        [~,useNum(ss,pp)] = find(contains(probeInfo.implantedSubjects{1},subjects{ss}));

        probeIdx = contains(probeSNAll,probes(pp));
        subAndProbeIdx = find(subjectIdx & probeIdx);
        recLocGood = recLocAll(subAndProbeIdx);

        recLoc = unique(recLocGood);
        for rr = 1:numel(recLoc)
            recIdx = find(strcmp(recLocAll,recLoc{rr}));
            if numel(unique(days(recIdx)))>2

                recLocSlope{ss,pp}{rr} = recLoc{rr};

                % Compute the slope
                X = [ones(numel(recIdx),1), days(recIdx)'];
                b{ss,pp}(rr,:) = (X\log10(clusterNum(recIdx)'));

                if pltIndiv
                    plot(days(recIdx), clusterNum(recIdx),'-','color',[colAni(ss,:) .2])
                    scatter(days(recIdx), clusterNum(recIdx),5,colAni(ss,:),'filled','MarkerEdgeAlpha',0.2,'MarkerFaceAlpha',0.2)
                    plot(days(recIdx), 10.^(X*b{ss,pp}(rr,:)'), 'color',colAni(ss,:),'LineWidth',1)
                end
            else
                recLocSlope{ss,pp}{rr} = '';
                b{ss,pp}(rr,:) = [nan;nan];
            end
        end

        % 'running average' for fullProbeScan?
        fullProbeScanSpec = cellfun(@(x) [subjects{ss} '__' probes{pp} '__' x{1}], fullProbeScan, 'uni', 0);
        ee = numel(fullProbeScanSpec)+2;
        n = 1;
        clear clusterNumProbe dayFullProbe
        while ee < numel(recLocGood)
            if all(cell2mat(cellfun(@(x) ismember(x,recLocGood(ee-numel(fullProbeScanSpec)-2+1:ee)), fullProbeScanSpec, 'uni', 0)))
                clusterNumProbe(n) = sum(clusterNum(subAndProbeIdx(ee-numel(fullProbeScanSpec)-2+1:ee)));
                dayFullProbe(n) = days(subAndProbeIdx(ee));
                ee = ee+numel(fullProbeScanSpec)+2;
                n = n+1;
            else
                ee = ee+1;
            end
        end
        if exist('clusterNumProbe','var')
            plot(dayFullProbe,clusterNumProbe,'-','color',[colAni(ss,:) .2])
            scatter(dayFullProbe,clusterNumProbe,15,colAni(ss,:),'filled','MarkerEdgeAlpha',0.5,'MarkerFaceAlpha',0.5)
            X = [ones(numel(dayFullProbe),1), dayFullProbe'];
            ball = (X\log10(clusterNumProbe'));
            plot(dayFullProbe, 10.^(X*ball), 'color',colAni(ss,:),'LineWidth',2)
            text(dayFullProbe(end), 10.^(X(end,:)*ball),[subjects{ss} ' ' probes{pp}],'color',colAni(ss,:))
        end

        slopeMean(ss,pp) = nanmean(b{ss,pp}(:,2));
    end
end

set(gca, 'YScale', 'log')
set(gca, 'XScale', 'log')
ylabel({'Unit count';''})
xlabel('Days from implantation')
yticks([1 10 100 1000])
yticklabels([1 10 100 1000])
xticks([5 10 25 50 100])
xticklabels([5 10 25 50 100])
ylim([10,2000])
xlim([3,max(days)])

% slope
figure('Position',[680   728   200   250]);
tmp = slopeMean(~isnan(slopeMean(:)));
uses = useNum(~isnan(slopeMean(:)));
colAnitmp = [colAni(~isnan(slopeMean(:,1)),:); colAni(~isnan(slopeMean(:,2)),:)];
[~,idx] = sort(tmp);
scatter(1:numel(tmp),100*(10.^(tmp(idx))-1),40*uses,colAnitmp(idx,:),'filled');
ylabel({'% change of unit';  ' count (%/day)'})
xlabel('Experiment')

%% ------ Natural images plots ------
%% Get data

mice = csv.readTable(csv.getLocation('main'));
[dball, res] = natim.main(subject=mice(contains(mice.Subject, 'CB'),:).Subject);
save('\\zserver\Lab\Share\Celian\dataForSfn2022_ChronicImplant_natIm','dball', 'res','-v7.3')

load('\\zserver\Lab\Share\Celian\dataForSfn2022_ChronicImplant_natIm')

%% Plot it

%% Get natural images
imgDir = '\\zserver.cortexlab.net\Data\pregenerated_textures\Anna\selection112';
imID = 1;
im = load(fullfile(imgDir, sprintf('img%d.mat',imID)));
figure;
imagesc(im.img)
colormap(gray)
axis equal tight

%% Plot stability
colAni = lines(numel(subjects));
figure;
hold all
for rr = 1:numel(recLocUni)
    subjectIdx = strcmp(subjects,subjectsAll{rr});
    plot(idi{rr},NstableDur{rr},'-','color',[colAni(subjectIdx,:)])
    scatter(idi{rr},NstableDur{rr},10,[colAni(subjectIdx,:)],'filled')
end
ylabel({'Number of'; 'matched clusters'})
xlabel('Number of days between recordings')
set(gca, 'YScale', 'log')
% xlim([0.5 7])
% xticks([1 7])
% xticklabels([1 7])
ylim([1 1000])
yticks([1 10 100 1000])
yticklabels([1 10 100 1000])

figure;
subplot(311); hold all
for rr = 1:numel(recLocUni)
    subjectIdx = strcmp(subjects,subjectsAll{rr});
    plot(idi{rr},PstableDur{rr}*100,'-','color',[colAni(subjectIdx,:) .2])
    scatter(idi{rr},PstableDur{rr}*100,5,[colAni(subjectIdx,:)],'filled','MarkerEdgeAlpha',0.2,'MarkerFaceAlpha',0.2)

    notnan = ~isnan(PstableDur{rr}); notnan(1) = 0;
    X = [ones(numel(idi{rr}(notnan)),1), log10(idi{rr}(notnan))];
    b = (X\PstableDur{rr}(notnan)'*100);
    plot(idi{rr}(2:end), X*b, 'color',colAni(subjectIdx,:),'LineWidth',2)
    text(idi{rr}(end), X(end,:)*b,[subjects(subjectIdx) '-' recLocUni{rr}],'color',colAni(subjectIdx,:))
end
ylim([0 100])
xticks([1 5 10 20 30])
xticklabels([1 5 10 20 30])
xlim([1,30])
ylabel({'Proportion of'; 'matched clusters'})
xlabel('Number of days between recordings')

subplot(312); hold all
for rr = 1:numel(recLocUni)
    plot(idi{rr},sigCorrStructDur{rr},'-','color',[colAni(strcmp(subjects,subjectsAll{rr}),:) .2])
    scatter(idi{rr},sigCorrStructDur{rr},5,colAni(strcmp(subjects,subjectsAll{rr}),:),'filled','MarkerEdgeAlpha',0.2,'MarkerFaceAlpha',0.2)

    notnan = ~isnan(sigCorrStructDur{rr}); notnan(1) = 0;
    X = [ones(numel(idi{rr}(notnan)),1), idi{rr}(notnan)];
    b = (X\sigCorrStructDur{rr}(notnan)');
    plot(idi{rr}(notnan), X*b, 'color',colAni(strcmp(subjects,subjectsAll{rr}),:),'LineWidth',2)
end
ylim([0 1])
xticks([1 5 10 20 30])
xticklabels([1 5 20 30])
xlim([1,30])
ylabel({'Stability of' ; 'signal correlations'})
xlabel('Number of days between recordings')

subplot(313); hold all
for rr = 1:numel(recLocUni)
    plot(idi{rr},noiseCorrStructDur{rr},'-','color',[colAni(strcmp(subjects,subjectsAll{rr}),:) .2])
    scatter(idi{rr},noiseCorrStructDur{rr},5,colAni(strcmp(subjects,subjectsAll{rr}),:),'filled','MarkerEdgeAlpha',0.2,'MarkerFaceAlpha',0.2)

    notnan = ~isnan(noiseCorrStructDur{rr}); notnan(1) = 0;
    X = [ones(numel(idi{rr}(notnan)),1), idi{rr}(notnan)];
    b = (X\noiseCorrStructDur{rr}(notnan)');
    plot(idi{rr}(notnan), X*b, 'color',colAni(strcmp(subjects,subjectsAll{rr}),:),'LineWidth',2)
end
ylim([0 1])
xticks([1 5 10 20 30])
xticklabels([1 5 10 20 30])
xlim([1,30])
ylabel({'Stability of' ; 'noise correlations'})

%% Plot correlation

expIdx2Keep = find(strcmp({dball.recLoc},recLocUni{1}));
db = dball(expIdx2Keep);

for dd = 1:numel(db)
    s = size(db(dd).spikeData);
    resp1 = reshape(nanmean(db(dd).spikeData(:,:,:,1:2:end),[1 4]), [s(2),s(3)]);
    resp2 = reshape(nanmean(db(dd).spikeData(:,:,:,2:2:end),[1 4]), [s(2),s(3)]);
    reliability = diag(corr(resp1,resp2));

    reliableUnits = reliability>0.3;
    db(dd).spikeData = db(dd).spikeData(:,:,reliableUnits,:);
    db(dd).C.XPos = db(dd).C.XPos(reliableUnits);
    db(dd).C.Depth = db(dd).C.Depth(reliableUnits);
    db(dd).C.CluID = db(dd).C.CluID(reliableUnits);
    db(dd).C.CluLab = db(dd).C.CluLab(reliableUnits);
end

% Perform CCA
[corrWCCA, ~] = natim.computeCCA({db.spikeData});

% Get the best matches
[BestMatch, BestCorr] = natim.getBestMatch(corrWCCA);

% Compute and correct for drift
Depth = cellfun(@(x) x.Depth, {db.C}, 'uni', 0);
DepthCorrected = natim.correctDrift(BestMatch, BestCorr, Depth);
for i = 1:numel(db); db(i).C.DepthCorrected = DepthCorrected{i}; end

% Get distance between clusters
XPos = cellfun(@(x) x.XPos, {db.C}, 'uni', 0);
BestDist = natim.getBestDist(BestMatch, XPos, DepthCorrected);

[pairAcrossAll_fewDays,sigCorr,noiseCorr] = natim.plotMatchedNeuronsAcrossDays(1:5, BestMatch, BestCorr, BestDist, ...
        {db.spikeData}, XPos, DepthCorrected, [db.days],[0.05 0.5 150]);