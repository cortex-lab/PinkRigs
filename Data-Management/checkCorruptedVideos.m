clear all
close all

%% load csv w/ corrupted videos

A = readtable('\\zserver\Code\AVrig\video_corruption_check.csv');

%% check properties of the corrupted videos

nExp = numel(A.expNum);
nRow = 3;
nCol = 5;

figure;

n = 1;
% rig?
rigList = unique(A.rigName);
for r = 1:numel(rigList)
    freqRig(r) = sum(strcmp(A.rigName,rigList{r}) & A.vid_corrupted == 1)/sum(strcmp(A.rigName,rigList{r}));
end
subplot(nRow,nCol,n)
bar(1:numel(rigList),freqRig)
xticks(1:numel(rigList))
xticklabels(rigList)
xtickangle(45)
xlabel('zelda')
ylabel('frequency of occurence')
n = n+1;

% camera?
fovList = unique(A.video_fov);
for r = 1:numel(fovList)
    freqCam(r) = sum(strcmp(A.video_fov,fovList{r}) & A.vid_corrupted == 1)/sum(strcmp(A.video_fov,fovList{r}));
end
subplot(nRow,nCol,n)
bar(1:numel(fovList),freqCam)
xticks(1:numel(fovList))
xticklabels(fovList)
xtickangle(45)
xlabel('camera')
ylabel('frequency of occurence')
n = n+1;

% exp length
dur = A.expDuration;
subplot(nRow,nCol,n)
scatter(A.vid_corrupted + 0.1*randn(nExp,1),dur)
ylim([0,prctile(dur,99)])
xticks(0:1)
xticklabels({'notcorr.', 'corr'})
xtickangle(45)
xlabel('status')
ylabel('duration')
n = n+1;

% frame rate? (just an estimate here -- can't have access to it because corrupted!)
fr = A.fileSizeBytes./A.expDuration;
subplot(nRow,nCol,n)
scatter(A.vid_corrupted + 0.1*randn(nExp,1),fr)
ylim([0,prctile(fr,99)])
xticks(0:1)
xticklabels({'notcorr.', 'corr'})
xtickangle(45)
xlabel('status')
ylabel('size/duration')
n = n+1;

% total size of the recordings for that exp
nn = 1;
ani = unique(A.subject);
clear corrRec sizeRec
for a = 1:numel(ani)
    dates = unique(A.expDate(strcmp(A.subject,ani{a})));
    for d = 1:numel(dates)
        expnum = unique(A.expNum(strcmp(A.subject,ani{a}) & A.expDate == dates(d)));
        for e = 1:numel(expnum)
            idx = find(strcmp(A.subject,ani{a}) & (A.expNum == expnum(e)) & (A.expDate == dates(d)));
            sizeRec(nn) = sum(A.fileSizeBytes(idx));
            corrRec(nn) = any(A.vid_corrupted(idx));
            frRec(idx) = sum(A.fileSizeBytes(idx))/A.expDuration(idx(1));
            nn = nn+1;
        end
    end
end
subplot(nRow,nCol,n)
scatter(corrRec' + 0.1*randn(numel(sizeRec),1),sizeRec')
ylim([0,prctile(sizeRec,99)])
xticks(0:1)
xticklabels({'notcorr.', 'corr'})
xtickangle(45)
xlabel('status')
ylabel('size of all videos')
n = n+1;

% total size of the recordings for that exp
nn = 1;
ani = unique(A.subject);
clear corrRec sizeRec
for a = 1:numel(ani)
    dates = unique(A.expDate(strcmp(A.subject,ani{a})));
    for d = 1:numel(dates)
        expnum = unique(A.expNum(strcmp(A.subject,ani{a}) & A.expDate == dates(d)));
        for e = 1:numel(expnum)
            idx = find(strcmp(A.subject,ani{a}) & (A.expNum == expnum(e)) & (A.expDate == dates(d)));
            frRec(idx) = sum(A.fileSizeBytes(idx))/A.expDuration(idx(1));
            nn = nn+1;
        end
    end
end
subplot(nRow,nCol,n)
scatter(A.vid_corrupted' + 0.1*randn(nExp,1),frRec')
ylim([0,prctile(sizeRec,99)])
xticks(0:1)
xticklabels({'notcorr.', 'corr'})
xtickangle(45)
xlabel('status')
ylabel('size of all videos')
n = n+1;

% how many cameras corrupted each time?
subplot(nRow,nCol,n)
nn = 1;
ani = unique(A.subject);
clear corrNum camNum
for a = 1:numel(ani)
    dates = unique(A.expDate(strcmp(A.subject,ani{a})));
    for d = 1:numel(dates)
        expnum = unique(A.expNum(strcmp(A.subject,ani{a}) & A.expDate == dates(d)));
        for e = 1:numel(expnum)
            
            idx = strcmp(A.subject,ani{a}) & (A.expNum == expnum(e)) & (A.expDate == dates(d));
            corrNum(nn) = sum(A.vid_corrupted(idx));
            camNum(nn) = sum(idx);
            nn = nn+1;
        end
    end
end
hold all
bar(1:3,[sum(corrNum==1), sum(corrNum==2), sum(corrNum==3)])
xticks(1:3)
xlabel('nb of corr cam')
ylabel('freq.')
n = n+1;

% per animal
ani = unique(A.subject);
for a = 1:numel(ani)
    freqAni(a) = sum(strcmp(A.subject,ani{a}) & A.vid_corrupted == 1)/sum(strcmp(A.subject,ani{a}));
end
[~,sortidx] = sort(freqAni);
subplot(nRow,nCol,n)
bar(1:numel(ani),freqAni(sortidx))
xticks(1:numel(ani))
xticklabels(ani(sortidx))
xtickangle(45)
xlabel('animal')
ylabel('frequency of occurence')
n = n+1;

% date?
dates = unique(A.expDate);
for d = 1:numel(dates)
    for r = 1:numel(rigList)
        idx = strcmp(A.rigName,rigList{r}) & A.expDate == dates(d);
        if sum(idx)>0
            freqDate(d,r) = sum(idx & A.vid_corrupted == 1)/sum(idx);
        else
            freqDate(d,r) = nan;
        end
    end
end
subplot(nRow,nCol,n)
hold all
datesnum = datenum(dates);
colRig = winter(4);
clear s
for r = 1:numel(rigList)
    s(r) = scatter(datesnum,freqDate(:,r), 30, colRig(r,:), 'filled');
end
xticks(linspace(datesnum(1),datesnum(end), 5))
xticklabels(datestr(linspace(datesnum(1),datesnum(end), 5),2))
xtickangle(45)
xlabel('date')
ylabel('frequency of occurence')
legend(s,rigList)
n = n+1;

% presence of the last frame?
subplot(nRow,nCol,n)
bar(1:2, [sum(A.vid_corrupted==1 & strcmp(A.lastFrameVidExist,'True'))/sum(A.vid_corrupted==1), ...
    sum(A.vid_corrupted==0 & strcmp(A.lastFrameVidExist,'True'))/sum(A.vid_corrupted==0)])
xticks(1:2)
xticklabels({'notcorr.', 'corr'})
xtickangle(45)
xlabel('status')
ylabel('freq. of last frames absence')
n = n+1;

% what type of expDef?
expDef = unique(A.expDef);
for a = 1:numel(expDef)
    freqExpDef(a) = sum(strcmp(A.expDef,expDef{a}) & A.vid_corrupted == 1)/sum(strcmp(A.expDef,expDef{a}));
end
[~,sortidx] = sort(freqExpDef);
subplot(nRow,nCol,n)
bar(1:numel(expDef),freqExpDef(sortidx))
xticks(1:numel(expDef))
xticklabels(expDef(sortidx))
xtickangle(45)
xlabel('animal')
ylabel('frequency of occurence')
n = n+1;

%% :)

Amdl = readtable('\\zserver\Code\AVrig\video_corruption_check.csv', 'TextType', 'string', 'DatetimeType', 'text');
Amdl = removevars(Amdl,{'Var1', ... % just the index
    'subject', ... % quite unlikely this is the issue
    'video_fpath', ... % maybe add server as a variable? but files are corrupted even locally
    'lastFrameVidExist'}); % super predictive but obvious
Amdl.expDef = string(A.expDef);
Amdl.fr = A.fileSizeBytes./A.expDuration;
Amdl.frTot = frRec';

% mdl = fitlm(Amdl,'ResponseVar','vid_corrupted');
% mdl.anova
% figure; plot(mdl)

mdl = fitglm(Amdl,'ResponseVar','vid_corrupted','Distribution','binomial','Link','logit');

figure;
idx = find(mdl.Coefficients.pValue<0.05);
[~,sortidx] = sort(mdl.Coefficients.pValue(idx),'ascend');
idx = idx(sortidx);
bar(1:numel(idx),mdl.Coefficients.pValue(idx))
set(gca,'Yscale','log')
xticks(1:numel(idx))
xticklabels(mdl.Coefficients.Properties.RowNames(idx))
xtickangle(45)
xlabel('coeff')
ylabel('pvalue')