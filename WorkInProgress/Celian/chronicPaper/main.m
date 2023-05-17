%% Code to reproduce the figures for the Chronic paper.

%% Get data

recompute = 1;
if recompute
    [clusterNum, recLocAll, days, expInfoAll] = plts.spk.clusterCount(pltIndiv=0,getQM=1,getPos=1);
    save('\\znas.cortexlab.net\Lab\Share\Celian\dataForPaper_ChronicImplant_stability_withQM','clusterNum', 'recLocAll', 'days', 'expInfoAll')
else
    load('\\znas.cortexlab.net\Lab\Share\Celian\dataForPaper_ChronicImplant_stability_withQM');
end
expInfoAll = cat(1,expInfoAll{:});

% Get mice info
mice = csv.readTable(csv.getLocation('main'));

recInfo = cellfun(@(x) split(x,'__'),recLocAll,'uni',0);
subjectsAll = cellfun(@(x) x{1}, recInfo, 'UniformOutput', false);
probeSNAll = cellfun(@(x) x{2}, recInfo, 'UniformOutput', false);

subjects = unique(subjectsAll);
colAni = lines(numel(subjects));

% Get probe info
probeSNUni = unique(probeSNAll);
probeInfo = csv.checkProbeUse(str2double(probeSNUni));

% Full scan info
fullProbeScan = {{'0__2880'}, {'1__2880'}, {'2__2880'}, {'3__2880'}, ...
    {'0__0'}, {'1__0'}, {'2__0'}, {'3__0'}};

% BC params
e.folder = ''; e.name = ''; % hack
paramBC = bc_qualityParamValues(e, '');

%% Plot depth raster for full probe

ss = find(contains(subjects,'AV009'));
pp = 1;

subjectIdx = contains(subjectsAll,subjects(ss));
probes = unique(probeSNAll(subjectIdx));

fullProbeScanSpec = cellfun(@(x) [subjects{ss} '__' probes{pp} '__' x{1}], fullProbeScan, 'uni', 0);

meas = cell(1,numel(fullProbeScanSpec));
for rr = 1:numel(fullProbeScanSpec)
    recIdx = find(strcmp(recLocAll,fullProbeScanSpec{rr}));
    daysSinceImplant{rr} = expInfoAll(recIdx,:).daysSinceImplant; % just to plot the same
    d = split(fullProbeScan{rr},'__');
    depthBins{rr} = str2num(d{2}) + (0:20:2880);
    meas{rr} = nan(numel(recIdx),numel(depthBins{rr})-1);
    for dd = 1:numel(recIdx)
        probeName = fieldnames(expInfoAll(recIdx(dd),:).dataSpikes{1});
        clusters = expInfoAll(recIdx(dd),:).dataSpikes{1}.(probeName{1}).clusters;
        for depth = 1:numel(depthBins{rr})-1
            depthNeuronIdx = (clusters.depths > depthBins{rr}(depth)) & (clusters.depths < depthBins{rr}(depth+1));
            unitType = bc_getQualityUnitType(paramBC, clusters.bc_qualityMetrics);
            meas{rr}(dd,depth) = sum(clusters.qualityMetrics.firing_rate(depthNeuronIdx & (unitType == 1)));
%             meas{rr}(dd,depth) = nanmean(clusters.qualityMetrics.amp_median(depthNeuronIdx & (unitType == 1)));
        end
    end
end

colors = winter(5);
figure('Position', [680   282   441   685]);
for rr = 1:numel(fullProbeScanSpec)
    ax(rr) = subplot(2,4,rr);
    imagesc(cell2mat(daysSinceImplant{rr}),depthBins{rr},meas{rr}')
    set(gca,'YDir','normal')
    c = [linspace(1,colors(mod(rr-1,4)+1,1),64)', linspace(1,colors(mod(rr-1,4)+1,2),64)', linspace(1,colors(mod(rr-1,4)+1,3),64)'];
    colormap(ax(rr),c)
    clim([0 20]);
end


%% Plot stability

qm = nan(1, size(expInfoAll,1));
for rr = 1:size(expInfoAll,1)
    probeName = fieldnames(expInfoAll(rr,:).dataSpikes{1});
    clusters = expInfoAll(rr,:).dataSpikes{1}.(probeName{1}).clusters;
    if ~isempty(clusters.qualityMetrics)
%         idx2Use = strcmp(clusters.qualityMetrics.ks2_label,"good");
        unitType = bc_getQualityUnitType(paramBC, clusters.bc_qualityMetrics);
        idx2Use = unitType == 1;

%         qm(rr) = sum(clusters.bc_qualityMetrics.nSpikes(idx2Use)); yRng = [100 20000]; %Total spks/s
%         qm(rr) = sum(clusters.qualityMetrics.firing_rate(idx2Use)); yRng = [100 20000]; %Total spks/s
%         qm(rr) = nanmedian(clusters.qualityMetrics.amp_median(idx2Use)); yRng = [100 200]; %Median spk amp
%         qm(rr) = nanmedian(clusters.qualityMetrics.missed_spikes_est(idx2Use));
        qm(rr) = sum(idx2Use); yRng = [10 2000]; %Total units


    else
        % happens a few times, will fix later
        qm(rr)= nan;
    end
end

slopeMean = nan(numel(subjects),2);
% figure
figure('Position',[680 728 400 300]);
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
        [~,useNum(ss,pp)] = find(contains(probeInfo.implantedSubjects{contains(probeSNUni,probes(pp))},subjects{ss}));

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
                b{ss,pp}(rr,:) = (X\log10(qm(recIdx)'));

                if pltIndiv
                    plot(days(recIdx), qm(recIdx),'-','color',[colAni(ss,:) .2])
                    scatter(days(recIdx), qm(recIdx),5,colAni(ss,:),'filled','MarkerEdgeAlpha',0.2,'MarkerFaceAlpha',0.2)
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
        clear qmProbe dayFullProbe
        while ee < numel(recLocGood)
            if all(cell2mat(cellfun(@(x) ismember(x,recLocGood(ee-numel(fullProbeScanSpec)-2+1:ee)), fullProbeScanSpec, 'uni', 0)))
                qmProbe(n) = sum(qm(subAndProbeIdx(ee-numel(fullProbeScanSpec)-2+1:ee)));
                dayFullProbe(n) = days(subAndProbeIdx(ee));
                ee = ee+numel(fullProbeScanSpec)+2;
                n = n+1;
            else
                ee = ee+1;
            end
        end
        if exist('qmProbe','var') && numel(dayFullProbe)>1
            plot(dayFullProbe,qmProbe,'-','color',[colAni(ss,:) .2])
            scatter(dayFullProbe,qmProbe,15,colAni(ss,:),'filled')
            X = [ones(numel(dayFullProbe),1), dayFullProbe'];
            ball = (X\log10(qmProbe'));
            plot(dayFullProbe, 10.^(X*ball), 'color',colAni(ss,:),'LineWidth',2)
            fullProbeSubj{end+1} = [subjects{ss} ' ' probes{pp}];
%             text(dayFullProbe(end), 10.^(X(end,:)*ball),fullProbeSubj{end},'color',colAni(ss,:))
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
ylim(yRng)
xlim([3,max(days)])
% offsetAxes

%% Summary plots

% slope
SteinmetzSlopes = 100*(10.^([-0.025 0.01])-1);
slopeVec = slopeMean(~isnan(slopeMean(:)));
uses = useNum(~isnan(slopeMean(:)));
subjVec = subj(~isnan(slopeMean(:))); % This is wrong
colAnitmp = [colAni(~isnan(slopeMean(:,1)),:); colAni(~isnan(slopeMean(:,2)),:)];
[~,idx] = sort(slopeVec);
x = 1:numel(slopeVec);
y = 100*(10.^(slopeVec)-1);
figure('Position',[680   727   404   251]);
ax(1) = subplot(121);
hold all
patch([min(x) min(x) max(x) max(x)], [SteinmetzSlopes SteinmetzSlopes(end:-1:1)], ones(1,3)*0.9,  'EdgeColor','none')
scatter(x,y(idx),40*uses(idx),[0.5 0.5 0.5],'filled');
fullProbeIdx = find(contains(subjVec(idx),fullProbeSubj));
scatter(x(fullProbeIdx),y(idx(fullProbeIdx)),40*uses(idx(fullProbeIdx)),colAnitmp(idx(fullProbeIdx),:),'filled');
ylabel({'% change of unit';  ' count (%/day)'})
xlabel('Experiment')
ax(2) = subplot(122); 
hold all
patch([0 0 5 5], [SteinmetzSlopes SteinmetzSlopes(end:-1:1)], ones(1,3)*0.9,  'EdgeColor','none')
h = histogram(y,min(y):1:max(y),'orientation','horizontal','EdgeColor','none','FaceColor',[.5 .5 .5]);
linkaxes(ax,'y')

% slope as a function of AP position
% find pos of each probe
probeRef = regexp(subjVec,' ','split');
APpos = nan(1,numel(probeRef));
MLpos = nan(1,numel(probeRef));
for p = 1:numel(probeRef)
    probeIdx = contains(probeSNUni,probeRef{p}{2});
    subIdx = strcmp(probeInfo.implantedSubjects{probeIdx},probeRef{p}{1});
    APpos(p) = str2double(probeInfo.positionAP{probeIdx}{subIdx});
    MLpos(p) = str2double(probeInfo.positionML{probeIdx}{subIdx});
end

% plot slope as a function of AP position
figure('Position',[680   728   200   250]);
hold all
[~,idx] = sort(APpos);
x = APpos;
y = 100*(10.^(slopeVec)-1);
patch([min(x) min(x) max(x) max(x)], [SteinmetzSlopes SteinmetzSlopes(end:-1:1)], ones(1,3)*0.9,  'EdgeColor','none')
scatter(x(idx),y(idx),40*uses(idx),[0.5 0.5 0.5],'filled');
fullProbeIdx = find(contains(subjVec(idx),fullProbeSubj));
scatter(x(idx(fullProbeIdx)),y(idx(fullProbeIdx)),40*uses(idx(fullProbeIdx)),colAnitmp(idx(fullProbeIdx),:),'filled');
ylabel({'% change of unit';  ' count (%/day)'})
xlabel('AP position')
offsetAxes

figure('Position',[680   728   200   250]);
hold all
[~,idx] = sort(APpos);
x = MLpos;
y = APpos;
scatter(x(idx),y(idx),40*uses(idx),[0.5 0.5 0.5],'filled');
fullProbeIdx = find(contains(subjVec(idx),fullProbeSubj));
scatter(x(idx(fullProbeIdx)),y(idx(fullProbeIdx)),40*uses(idx(fullProbeIdx)),colAnitmp(idx(fullProbeIdx),:),'filled');
ylabel('AP position')
xlabel('ML position')
axis equal tight
offsetAxes

figure('Position',[680   728   200   250]);
x = uses;
y = 100*(10.^(slopeVec)-1);
scatter(x,y,40*uses,[0.5 0.5 0.5],'filled');
scatter(x,y,40*uses,colAnitmp,'filled');
ylabel({'% change of unit';  ' count (%/day)'})
xlabel('Experiment')
offsetAxes
