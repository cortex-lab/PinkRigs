%% ------ Stability plots ------
%% Get data

% [clusterNum, recLocAll, days] = plt.spk.clusterCount(pltIndiv=0);
% save('\\zserver\Lab\Share\Celian\dataForSfn2022_ChronicImplant_stability','clusterNum', 'recLocAll', 'days')

load('\\zserver\Lab\Share\Celian\dataForSfn2022_ChronicImplant_stability')

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
fullProbeSubj = {};
subj = {};
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
        if exist('clusterNumProbe','var') && numel(dayFullProbe)>1
            plot(dayFullProbe,clusterNumProbe,'-','color',[colAni(ss,:) .2])
            scatter(dayFullProbe,clusterNumProbe,15,colAni(ss,:),'filled','MarkerEdgeAlpha',0.5,'MarkerFaceAlpha',0.5)
            X = [ones(numel(dayFullProbe),1), dayFullProbe'];
            ball = (X\log10(clusterNumProbe'));
            plot(dayFullProbe, 10.^(X*ball), 'color',colAni(ss,:),'LineWidth',2)
            fullProbeSubj{end+1} = [subjects{ss} ' ' probes{pp}];
            text(dayFullProbe(end), 10.^(X(end,:)*ball),fullProbeSubj{end},'color',colAni(ss,:))
        end

        slopeMean(ss,pp) = nanmean(b{ss,pp}(:,2));
        subj{ss,pp} = [subjects{ss} ' ' probes{pp}];
    end
end
subj(cell2mat(cellfun(@(x) isempty(x), subj, 'uni', 0))) = {' '};

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
hold all
slopeVec = slopeMean(~isnan(slopeMean(:)));
uses = useNum(~isnan(slopeMean(:)));
subjVec = subj(~isnan(slopeMean(:))); % This is wrong
colAnitmp = [colAni(~isnan(slopeMean(:,1)),:); colAni(~isnan(slopeMean(:,2)),:)];
[~,idx] = sort(slopeVec);
x = 1:numel(slopeVec);
y = 100*(10.^(slopeVec)-1);
scatter(x,y(idx),40*uses(idx),[0.5 0.5 0.5],'filled');
fullProbeIdx = find(contains(subjVec(idx),fullProbeSubj));
scatter(x(fullProbeIdx),y(idx(fullProbeIdx)),40*uses(idx(fullProbeIdx)),colAnitmp(idx(fullProbeIdx),:),'filled');
ylabel({'% change of unit';  ' count (%/day)'})
xlabel('Experiment')

% slope as a function of AP position
% find pos of each probe
mice = csv.readTable(csv.getLocation('main'));
probeRef = regexp(subjVec,' ','split');
APpos = nan(1,numel(probeRef));
MLpos = nan(1,numel(probeRef));
for p = 1:numel(probeRef)
    probeInfo = csv.checkProbeUse(str2num(probeRef{p}{2}));
    subIdx = strcmp(probeInfo.implantedSubjects{1},probeRef{p}{1});
    APpos(p) = str2double(probeInfo.positionAP{1}{subIdx});
    MLpos(p) = str2double(probeInfo.positionML{1}{subIdx});
end

% plot slope as a function of AP position
figure('Position',[680   728   200   250]);
hold all
[~,idx] = sort(APpos);
x = APpos(idx);
y = 100*(10.^(slopeVec)-1);
scatter(x,y(idx),40*uses(idx),[0.5 0.5 0.5],'filled');
fullProbeIdx = find(contains(subjVec(idx),fullProbeSubj));
scatter(x(fullProbeIdx),y(idx(fullProbeIdx)),40*uses(idx(fullProbeIdx)),colAnitmp(idx(fullProbeIdx),:),'filled');
ylabel({'% change of unit';  ' count (%/day)'})
xlabel('AP position')

%% ------ Q metrics stability plots ------
%% Get data


%% Plot it


%% ------ Natural images plots ------
%% Get data

% mice = csv.readTable(csv.getLocation('main'));
% [dball, res] = natim.main(subject=mice(contains(mice.Subject, 'CB020'),:).Subject);
% save('\\zserver\Lab\Share\Celian\dataForSfn2022_ChronicImplant_natIm','dball', 'res','-v7.3')

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
subjects = unique(res.subjectsAll);
recLocUni = unique({dball.recLoc});
recLocUni = recLocUni(cell2mat(cellfun(@(x) sum(strcmp({dball.recLoc},x))>1, recLocUni, 'uni', 0)));
colAni = lines(numel(subjects));

% Starting from d1 only
figure;
hold all
for rr = 1:numel(recLocUni)
    subjectIdx = strcmp(subjects,res.subjectsAll{rr});
    plot(res.dur{rr}(:,1),res.Nstable{rr}(:,1),'-','color',[colAni(subjectIdx,:)])
    scatter(res.dur{rr}(:,1),res.Nstable{rr}(:,1),10,[colAni(subjectIdx,:)],'filled')
end
ylabel({'Number of'; 'matched clusters'})
xlabel('Number of days after exp start')
set(gca, 'YScale', 'log')
ylim([1 1000])
yticks([1 10 100 1000])
yticklabels([1 10 100 1000])

% All days
figure;
hold all
for rr = 1:numel(recLocUni)
    subjectIdx = strcmp(subjects,res.subjectsAll{rr});
    plot(res.idi{rr},res.NstableDur{rr},'-','color',[colAni(subjectIdx,:)])
    scatter(res.idi{rr},res.NstableDur{rr},10,[colAni(subjectIdx,:)],'filled')
end
ylabel({'Number of'; 'matched clusters'})
xlabel('Number of days between recordings')
set(gca, 'YScale', 'log')
ylim([1 1000])
yticks([1 10 100 1000])
yticklabels([1 10 100 1000])

figure;
subplot(311); hold all
for rr = 1:numel(recLocUni)
    subjectIdx = strcmp(subjects,res.subjectsAll{rr});
    plot(res.idi{rr},res.PstableDur{rr}*100,'-','color',[colAni(subjectIdx,:) .2])
    scatter(res.idi{rr},res.PstableDur{rr}*100,5,[colAni(subjectIdx,:)],'filled','MarkerEdgeAlpha',0.2,'MarkerFaceAlpha',0.2)

    notnan = ~isnan(res.PstableDur{rr}); notnan(1) = 0;
    X = [ones(numel(res.idi{rr}(notnan)),1), log10(res.idi{rr}(notnan))];
    b = (X\res.PstableDur{rr}(notnan)'*100);
    plot(res.idi{rr}(2:end), X*b, 'color',colAni(subjectIdx,:),'LineWidth',2)
    text(res.idi{rr}(end), X(end,:)*b,[subjects(subjectIdx) '-' recLocUni{rr}],'color',colAni(subjectIdx,:))
end
ylim([0 100])
xticks([1 5 10 20 30])
xticklabels([1 5 10 20 30])
xlim([1,30])
ylabel({'Proportion of'; 'matched clusters'})
xlabel('Number of days between recordings')

subplot(312); hold all
for rr = 1:numel(recLocUni)
    plot(res.idi{rr},res.sigCorrStructDur{rr},'-','color',[colAni(strcmp(subjects,res.subjectsAll{rr}),:) .2])
    scatter(res.idi{rr},res.sigCorrStructDur{rr},5,colAni(strcmp(subjects,res.subjectsAll{rr}),:),'filled','MarkerEdgeAlpha',0.2,'MarkerFaceAlpha',0.2)

    notnan = ~isnan(res.sigCorrStructDur{rr}); notnan(1) = 0;
    X = [ones(numel(res.idi{rr}(notnan)),1), res.idi{rr}(notnan)];
    b = (X\res.sigCorrStructDur{rr}(notnan)');
    plot(res.idi{rr}(notnan), X*b, 'color',colAni(strcmp(subjects,res.subjectsAll{rr}),:),'LineWidth',2)
end
ylim([0 1])
xticks([1 5 10 20 30])
xticklabels([1 5 20 30])
xlim([1,30])
ylabel({'Stability of' ; 'signal correlations'})
xlabel('Number of days between recordings')

subplot(313); hold all
for rr = 1:numel(recLocUni)
    plot(res.idi{rr},res.noiseCorrStructDur{rr},'-','color',[colAni(strcmp(subjects,res.subjectsAll{rr}),:) .2])
    scatter(res.idi{rr},res.noiseCorrStructDur{rr},5,colAni(strcmp(subjects,res.subjectsAll{rr}),:),'filled','MarkerEdgeAlpha',0.2,'MarkerFaceAlpha',0.2)

    notnan = ~isnan(res.noiseCorrStructDur{rr}); notnan(1) = 0;
    X = [ones(numel(res.idi{rr}(notnan)),1), res.idi{rr}(notnan)];
    b = (X\res.noiseCorrStructDur{rr}(notnan)');
    plot(res.idi{rr}(notnan), X*b, 'color',colAni(strcmp(subjects,res.subjectsAll{rr}),:),'LineWidth',2)
end
ylim([0 1])
xticks([1 5 10 20 30])
xticklabels([1 5 10 20 30])
xlim([1,30])
ylabel({'Stability of' ; 'noise correlations'})

%% Plot example recording

expIdx2Keep = find(strcmp({dball.recLoc},recLocUni{5}));
db = dball(expIdx2Keep);

for dd = 1:numel(db)
    s = size(db(dd).spikeData);
    resp1 = reshape(nanmean(db(dd).spikeData(:,:,:,1:2:end),[1 4]), [s(2),s(3)]);
    resp2 = reshape(nanmean(db(dd).spikeData(:,:,:,2:2:end),[1 4]), [s(2),s(3)]);
    reliability = diag(corr(resp1,resp2));

    reliableUnits = reliability>0.5;
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

[Nstable, Pstable, dur, NstableDur, PstableDur, idi, pairAcrossAll_pairsOfDays] = natim.getMatchingStability(BestMatch,BestCorr,BestDist,[db.days],[0.05 0.5 150],1);
figure; 
plot(dur,Nstable)

[pairAcrossAll_fewDays,sigCorr,noiseCorr] = natim.plotMatchedNeuronsAcrossDays(1:7, BestMatch, BestCorr, BestDist, ...
        {db.spikeData}, XPos, DepthCorrected, [db.days],[0.05 0.5 150],9);