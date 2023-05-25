%% Code to reproduce the figures for the Chronic paper.

%% Get data

recompute = 0;
if recompute
    [clusterNum, recLocAll, days, expInfoAll] = plts.spk.clusterCount(pltIndiv=0,getQM=1,getPos=1);
    expInfoAll = cat(1,expInfoAll{:});
    [clusterNum_add, recLocAll_add, days_add, expInfoAll_add] = clusterCount_loadNonZeldaSubject({'EB019','CB015'},1,1);
    clusterNum = cat(2,clusterNum,clusterNum_add);
    recLocAll = cat(2,recLocAll,recLocAll_add);
    days = cat(2,days,days_add);
    expInfoAll_add = cat(1,expInfoAll_add{:});
    expInfoAll_tmp = cell2table(cell(size(expInfoAll_add,1),size(expInfoAll,2)));
    expInfoAll_tmp.Properties.VariableNames = expInfoAll.Properties.VariableNames;
    expInfoAll_tmp.dataSpikes = expInfoAll_add.dataSpikes;
    expInfoAll_tmp.daysSinceImplant = num2cell(expInfoAll_add.daysSinceImplant);
    expInfoAll = cat(1,expInfoAll,expInfoAll_tmp);
    save('\\znas.cortexlab.net\Lab\Share\Celian\dataForPaper_ChronicImplant_stability_withQM','clusterNum', 'recLocAll', 'days', 'expInfoAll')
else
    load('\\znas.cortexlab.net\Lab\Share\Celian\dataForPaper_ChronicImplant_stability_withQM');
end

% Get mice info
mice = csv.readTable(csv.getLocation('main'));

recInfo = cellfun(@(x) split(x,'__'),recLocAll,'uni',0);
subjectsAll = cellfun(@(x) x{1}, recInfo, 'UniformOutput', false);
probeSNAll = cellfun(@(x) x{2}, recInfo, 'UniformOutput', false);

subjects = unique(subjectsAll);
colAni = hsv(numel(subjects));

% Get probe info
probeSNUni = unique(probeSNAll);
probeInfo = csv.checkProbeUse(str2double(probeSNUni));
markersProbes = ["o","+","*",".","x","_","|","square","diamond","^","v",">","<","pentagram","hexagram", ...
    "o","square","diamond","^","v",">","<","pentagram","hexagram"];
filledProbes = [ones(1,15), zeros(1,9)];
markersProbes = markersProbes(1:numel(probeSNUni));
filledProbes = filledProbes(1:numel(probeSNUni));
% colProbes = winter(numel(probeSNUni));

% Full scan info
fullProbeScan = {{'0__2880'}, {'1__2880'}, {'2__2880'}, {'3__2880'}, ...
    {'0__0'}, {'1__0'}, {'2__0'}, {'3__0'}};

% BC params
e.folder = ''; e.name = ''; % hack
paramBC = bc_qualityParamValues(e, '');

%% Plot depth raster for full probe

ss = find(contains(subjects,'AV009'));
pp = 2;

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

fun = @(x) sqrt(x);
colors = winter(5);
figure('Position', [680   282   441   685], 'Name', [subjects{ss} '__' probes{pp}]);
for rr = 1:numel(fullProbeScanSpec)
    ax(rr) = subplot(2,4,rr);
    imagesc(cell2mat(daysSinceImplant{rr}),depthBins{rr},fun(meas{rr}'))
    set(gca,'YDir','normal')
    % c = [linspace(1,colors(mod(rr-1,4)+1,1),64)', linspace(1,colors(mod(rr-1,4)+1,2),64)', linspace(1,colors(mod(rr-1,4)+1,3),64)'];
    % colormap(ax(rr),c)
    c = colormap("gray"); c = flipud(c);
    colormap(c)
    clim([0 fun(20)]);
end

% anat
% plotAtlasSliceSchematics(630,[],8,[],[])
% plotAtlasSliceSchematics(470,[],8,[],[])

%% Plot raw traces?

% use zmat?

%% Plot the distributions of spikes amplitudes across days

ampBinEdges = 10.^(-5:0.01:-2);
ampBins = ampBinEdges(1:end-1)*10.^0.005;
H = nan(numel(ampBinEdges)-1,max(days(subjectIdx))+1,numel(fullProbeScanSpec));
for rr = 1:numel(fullProbeScanSpec)
    recIdx = strcmp(recLocAll,fullProbeScanSpec{rr});
    daysUni = unique(days(recIdx));
    for dd = 1:numel(daysUni)
        recDayIdx = find(recIdx &  days == daysUni(dd),1);
        probeName = fieldnames(expInfoAll(rr,:).dataSpikes{1});
        spk = csv.loadData(expInfoAll(recDayIdx,:), dataType=probeName{1}, object='spikes');
        H(:,daysUni(dd)+1,rr) = histcounts(spk.dataSpikes{1}.(probeName{1}).spikes.amps,ampBinEdges);
    end
end

figure;
colors_time = winter(max(days(subjectIdx)+1));
for rr = 1:numel(fullProbeScanSpec)
    subplot(2,4,rr)
    hold all
    for dd = 1:size(H,2)
        if ~isnan(H(1,dd,rr))
            plot(ampBins,H(:,dd,rr)./sum(H(:,dd,rr)),'color',colors_time(dd,:))
        end
    end
    set(gca, 'XScale', 'log')
    xlim([10.^-5 10.^-2])
end

figure;
subplot(211)
hold all
for dd = 1:size(H,2)
    plot(ampBins,nansum(H(:,dd,1:4)./sum(H(:,dd,1:4),1),3),'color',colors_time(dd,:))
end
set(gca, 'XScale', 'log')
xlim([10.^-5 10.^-2])
title('Upper bank')
subplot(212)
hold all
for dd = 1:size(H,2)
    plot(ampBins,nansum(H(:,dd,5:8)./sum(H(:,dd,5:8),1),3),'color',colors_time(dd,:))
end
set(gca, 'XScale', 'log')
xlim([10.^-5 10.^-2])
title('Lower bank')

%% Extract metric

qm = nan(1, size(expInfoAll,1));
for rr = 1:size(expInfoAll,1)
    probeName = fieldnames(expInfoAll(rr,:).dataSpikes{1});
    clusters = expInfoAll(rr,:).dataSpikes{1}.(probeName{1}).clusters;
    if ~isempty(clusters.qualityMetrics)
        idx2Use = strcmp(clusters.qualityMetrics.ks2_label,"good");
        unitType = bc_getQualityUnitType(paramBC, clusters.bc_qualityMetrics);
        idx2Use = unitType == 1;

%         qm(rr) = sum(clusters.bc_qualityMetrics.nSpikes(idx2Use)); yRng = [100 20000]; %Total spks/s
%         qm(rr) = sum(clusters.qualityMetrics.firing_rate(idx2Use)); yRng = [100 20000]; %Total spks/s
%         qm(rr) = nanmedian(clusters.qualityMetrics.amp_median(idx2Use)); yRng = [100 200]; %Median spk amp
%         qm(rr) = nanmedian(clusters.qualityMetrics.missed_spikes_est(idx2Use));
        qm(rr) = sum(idx2Use); yRng = [1 4000]; %Total units

%           qm(rr) = clusterNum(rr);
    else
        % happens a few times, will fix later
        qm(rr)= nan;
    end
end

%% Plot stability

% figure
figure('Position',[600 500 360 230]);
hold all

% Find those that match location
% Do it by subject and probe so that it's easier to do the whole probe
% thing...?
% subjectsToInspect = {'AV009'};
subjectsToInspect = subjects;
colAniToInspect = colAni(ismember(subjects,subjectsToInspect),:);
dlim = 2;
pltIndivBank = 0;
pltIndivProbe = 0;
pltAllProbes = 1;

recLocSlope = cell(1,1);
b = cell(1,1);
slopeMean = nan(numel(subjectsToInspect),2);
useNum = nan(numel(subjectsToInspect),2);
fullProbeSubj = {};
subj = {};
daysSub = unique(days);
qmProbe = nan(numel(daysSub),2,numel(subjectsToInspect));
for ss = 1:numel(subjectsToInspect)
    subjectIdx = contains(subjectsAll,subjectsToInspect(ss));
    probes = unique(probeSNAll(subjectIdx));
    for pp = 1:numel(probes)

        % Check number of uses for this probe
        [~,useNum(ss,pp)] = find(contains(probeInfo.implantedSubjects{contains(probeSNUni,probes(pp))},subjectsToInspect{ss}));

        probeIdx = contains(probeSNAll,probes(pp));
        subAndProbeIdx = find(subjectIdx & probeIdx);
        recLocGood = recLocAll(subAndProbeIdx);

        recLoc = unique(recLocGood);
        for rr = 1:numel(recLoc)
            recIdx = find(strcmp(recLocAll,recLoc{rr}));
            if numel(unique(days(recIdx)))>1 && qm(recIdx(1)) > 10

                recLocSlope{ss,pp}{rr} = recLoc{rr};

                % Compute the slope
                X = [ones(numel(recIdx),1), days(recIdx)'];
                b{ss,pp}(rr,:) = (X\log10(qm(recIdx)'));

                if pltIndivBank && pp == 1
                    plot(days(recIdx), qm(recIdx),'-','color',[colAniToInspect(ss,:) .2])
                    scatter(days(recIdx), qm(recIdx),5,colAniToInspect(ss,:),'filled','MarkerEdgeAlpha',0.2,'MarkerFaceAlpha',0.2)
                    plot(days(recIdx), 10.^(X*b{ss,pp}(rr,:)'), 'color',colAniToInspect(ss,:),'LineWidth',1)
                end
            else
                recLocSlope{ss,pp}{rr} = '';
                b{ss,pp}(rr,:) = [nan;nan];
            end
        end

        slopeMean(ss,pp) = nanmean(b{ss,pp}(:,2));
        subj{ss,pp} = [subjectsToInspect{ss} ' ' probes{pp}];

        fullProbeScanSpec = cellfun(@(x) [subjectsToInspect{ss} '__' probes{pp} '__' x{1}], fullProbeScan, 'uni', 0);
        for dd = 1:numel(daysSub)
            day = daysSub(dd);
            % Find recordings around that date
            surrDaysIdx = find(abs(days(subAndProbeIdx) - day) <= dlim);
            [~,daysOrd] = sort(abs(days(subAndProbeIdx(surrDaysIdx))-day), 'ascend');
            scanIdx = cell2mat(cellfun(@(x) ismember(recLocGood(surrDaysIdx(daysOrd)),x)', fullProbeScanSpec, 'uni', 0));
            if all(sum(scanIdx,1))
                [~,scanIdx]=max(scanIdx,[],1);
                qmProbe(dd,pp,ss) = sum(qm(subAndProbeIdx(surrDaysIdx(daysOrd(scanIdx)))));

                % sanity check
                if ~all(cell2mat(cellfun(@(x) ismember(x,recLocGood(surrDaysIdx((daysOrd(scanIdx)))))', fullProbeScanSpec, 'uni', 0)))
                    error('problem with scan')
                end
            end
        end

        if pltIndivProbe
            % Show only one probe
            nanday = isnan(qmProbe(:,pp,ss));
            plot(daysSub(~nanday),qmProbe(~nanday,pp,ss),'-','color',[colAniToInspect(ss,:) .2])
            scatter(daysSub(~nanday),qmProbe(~nanday,pp,ss),15,colAniToInspect(ss,:),'filled')
            X = [ones(numel(daysSub(~nanday)),1), daysSub(~nanday)'];
            ball = (X\log10(qmProbe(~nanday,pp,ss)));
            plot(daysSub(~nanday), 10.^(X*ball), 'color',colAniToInspect(ss,:),'LineWidth',2)
            fullProbeSubj{end+1} = [subjectsToInspect{ss} ' ' probes{pp}];
            %             text(dayFullProbe(end), 10.^(X(end,:)*ball),fullProbeSubj{end},'color',colAniToInspect(ss,:))
        end
    end

    if pltAllProbes
        % Show two probes
        qmAllProbes = sum(qmProbe(:,:,ss),2);
        nanday = isnan(qmAllProbes);
        plot(daysSub(~nanday),qmAllProbes(~nanday),'-','color',[colAniToInspect(ss,:) .2])
        scatter(daysSub(~nanday),qmAllProbes(~nanday),30,colAniToInspect(ss,:),'filled')
        X = [ones(numel(daysSub(~nanday)),1), daysSub(~nanday)'];
        ball = (X\log10(qmAllProbes(~nanday)));
        plot(daysSub(~nanday), 10.^(X*ball), 'color',colAniToInspect(ss,:),'LineWidth',3)
        fullProbeSubj{end+1} = [subjectsToInspect{ss} ' ' probes{pp}];
    end
end
subj(cell2mat(cellfun(@(x) isempty(x), subj, 'uni', 0))) = {' '};

% set(gca, 'YScale', 'log')
% set(gca, 'XScale', 'log')
ylabel({'Unit count';''})
xlabel('Days from implantation')
yticks([1 10 100 1000])
yticklabels([1 10 100 1000])
xticks([5 10 25 50 100])
xticklabels([5 10 25 50 100])
ylim(yRng)
xlim([3,max(days)])
% offsetAxes

%% Summary quantif

% slope
SteinmetzSlopes = 100*(10.^([-0.025 0.01])-1);
slopeVec = slopeMean(~isnan(slopeMean(:)));
slopeVec = 100*(10.^(slopeVec)-1);
uses = useNum(~isnan(slopeMean(:)));
subjVec = subj(~isnan(slopeMean(:))); % Is this wrong?
probesVec = cellfun(@(y) y{2}, cellfun(@(x) strsplit(x,' '), subjVec, 'uni', 0), 'uni', 0);

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

% Mixed effects linear models
T = struct();
T.slope = slopeVec;
T.probeID = cell2mat(probesVec);
T.APpos = APpos';
T.MLpos = MLpos';
T.uses = uses;
T = struct2table(T);
fnames = T.Properties.VariableNames; 
fnames(contains(fnames,'slope')) = [];
fnames(contains(fnames,'probeID')) = [];
formula = 'slope ~ 1+';
for ff = 1:numel(fnames)
    formula = [formula fnames{ff} '+'];
end
formula = [formula '(1|probeID)'];
lme = fitlme(T,formula);

figure;
idx = find(lme.Coefficients.pValue<0.05);
[~,sortidx] = sort(lme.Coefficients.pValue(idx),'ascend');
idx = idx(sortidx);
bar(1:numel(idx),lme.Coefficients.pValue(idx))
set(gca,'Yscale','log')
xticks(1:numel(idx))
xticklabels(lme.CoefficientNames(idx))
xtickangle(45)
xlabel('coeff')
ylabel('pvalue')


%% Summary plots
colAnitmp = [colAniToInspect(~isnan(slopeMean(:,1)),:); colAniToInspect(~isnan(slopeMean(:,2)),:)];
[~,idx] = sort(slopeVec);
x = 1:numel(slopeVec);
y = slopeVec;
figure('Position',[680   727   404   180]);
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
patch([0 0 10 10], [SteinmetzSlopes SteinmetzSlopes(end:-1:1)], ones(1,3)*0.9,  'EdgeColor','none')
h = histogram(y(idx),linspace(min(y),max(y),20),'orientation','horizontal','EdgeColor','none','FaceColor',[.5 .5 .5]);
linkaxes(ax,'y')

% plot slope as a function of AP position
probesIdx = cell2mat(cellfun(@(x) contains(probesSNUni, x), probesVec, 'uni' ,0))
figure('Position',[680   728   200   180]); hold all
[~,idx] = sort(APpos);
x = APpos;
y = slopeVec;
patch([min(x) min(x) max(x) max(x)], [SteinmetzSlopes SteinmetzSlopes(end:-1:1)], ones(1,3)*0.9,  'EdgeColor','none')
scatter(x(idx),y(idx),40,[0.5 0.5 0.5],'filled');
fullProbeIdx = find(contains(subjVec(idx),fullProbeSubj));
scatter(x(idx(fullProbeIdx)),y(idx(fullProbeIdx)),40,colAnitmp(idx(fullProbeIdx),:),'filled','Markers',markersProbes());
% plot(unique(x),coeff(1)+coeff(2)*unique(x)+coeff(3)*nanmean(MLpos)+coeff(4)*nanmean(uses),'k')
ylabel({'% change of unit';  ' count (%/day)'})
xlabel('AP position')
offsetAxes

figure('Position',[680   728   200   180]); hold all
x = uses;
y = slopeVec;
probesVecUni = unique(probesVec);
for pp = 1:numel(probesVecUni)
    probeIdx = find(contains(probesVec,probesVecUni{pp}));
    [m,sortIdx] = sort(uses(probeIdx));
    if ~all(diff(m) == 1)
        error('probe use missing')
    end
    probeIdx = probeIdx(sortIdx);
    probeRef = contains(probeSNUni,probesVecUni{pp});
    scatter(x(probeIdx),y(probeIdx),30,'MarkerEdgeColor',colProbes(probeRef,:), ...
        'MarkerFaceColor',filledProbes(probeRef)*colProbes(probeRef,:)+(1-filledProbes(probeRef)), ...
        'Marker',markersProbes{probeRef});
    plot(x(probeIdx),y(probeIdx),'color',colProbes(probeRef,:));
end
% plot(unique(x),coeff(1)+coeff(2)*nanmean(APpos)+coeff(3)*nanmean(MLpos)+coeff(4)*unique(x),'k')
ylabel({'% change of unit';  ' count (%/day)'})
xlabel('Probe use')
offsetAxes

%% Compute slopes for cortical / subcortical
