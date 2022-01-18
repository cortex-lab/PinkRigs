function checkForNewAVRecordings(recompile)

serverLocations = { ...
    '\\znas.cortexlab.net\Subjects\'; ...
    '\\zubjects.cortexlab.net\Subjects\'; ...
    '\\128.40.224.65\Subjects\'; ...
    '\\zinu.cortexlab.net\Subjects\'};


if ~exist('recompile', 'var'); recompile = 0; end
csvLocation = '\\zserver.cortexlab.net\Code\AVrig\aMasterMouseList.csv';
csvData = readtable(csvLocation);
nanData = any(isnan(csvData.IsActive));

csvData.IsActive(isnan(csvData.IsActive)) = 0;
csvDataSort = sortrows(csvData, 'Subject', 'ascend');
csvDataSort = sortrows(csvDataSort, 'IsActive', 'descend');
if any(~strcmp(csvDataSort.Subject, csvData.Subject)) || nanData
    csvBackup(csvLocation);
    csvWriteClean(csvDataSort, csvLocation)
    csvData = csvDataSort;
end

if recompile 
    cycles = 2;
    mice2Update = csvData.Subject;
    paths2Check = cellfun(@(y) cellfun(@(x) [y x], mice2Update, 'uni', 0), serverLocations, 'uni', 0);
    paths2Check = vertcat(paths2Check{:});    
else
    cycles = 1;
    mice2Update = csvData.Subject(csvData.IsActive>0);
    paths2Check = cellfun(@(y) cellfun(@(x) [y x], mice2Update, 'uni', 0), serverLocations, 'uni', 0);
    
    past10Days = arrayfun(@(x) datestr(x, 'yyyy-mm-dd'), now-9:now, 'uni', 0)';
    paths2Check = cellfun(@(y) cellfun(@(x) [y filesep x], past10Days, 'uni', 0), vertcat(paths2Check{:}), 'uni', 0);
    paths2Check = vertcat(paths2Check{:});
end

for i = 1:cycles
    if isempty(paths2Check); continue; end
    fprintf('Detecting folder level %d ... \n', i);
    paths2Check = cellfun(@(x) java.io.File(x), paths2Check, 'uni', 0);
    paths2Check = cellfun(@(x) arrayfun(@char,x.listFiles,'uni',0), paths2Check, 'uni', 0);
    paths2Check = vertcat(paths2Check{:});
    paths2Check = paths2Check(~contains(paths2Check, {'Lightsheet';'ephys';'Backup';'g0'},'IgnoreCase',true));
    finalDigits = cellfun(@(x) length(x)-max(strfind(x, filesep)), paths2Check);
    if cycles - i == 1
        paths2Check(finalDigits~=10) = [];
    elseif cycles-i == 0
        paths2Check(finalDigits>2) = [];
    end
    paths2Check(isnan(cellfun(@(x) str2double(x(end)), paths2Check))) = [];
end

pathInfo = cellfun(@(x) split(x,filesep), paths2Check, 'uni', 0);
subList = cellfun(@(x) x{end-2}, pathInfo, 'uni', 0);
%%
for subject = mice2Update'
    currSub = subject{1};
    csvPathMouse = [fileparts(csvLocation) filesep currSub '.csv'];
    
    newDat.expDate = {};
    newDat.expNum = [];
    newDat.expDef = {};
    newDat.expDuration = [];
    newDat.rigName = {};
    newDat.ephys = [];
    newDat.frontCam = [];
    newDat.sideCam = [];
    newDat.eyeCam = [];    
    newDat.micDat = [];
    newDat.path = {};
    
    if ~exist(csvPathMouse, 'file')
        csvDataMouse = struct2table(newDat);
        writetable(csvDataMouse,csvPathMouse,'Delimiter',',');
    end
    
    currIdx = contains(subList, currSub);
    dateList = cellfun(@(x) x{end-1}, pathInfo(currIdx), 'uni', 0);
    expNumList = cellfun(@(x) x{end}, pathInfo(currIdx), 'uni', 0);
    nameStub = cellfun(@(x) [x{end-1} '_' x{end} '_' x{end-2}], pathInfo(currIdx), 'uni', 0);
    blockPath = cellfun(@(x,y) [x filesep y '_Block.mat'], paths2Check(currIdx), nameStub, 'uni', 0);
    
    csvDataMouse = readtable(csvPathMouse);
    
    if ~isempty(csvDataMouse.expDate)
        expRef = cellfun(@(x,y) [x,y], dateList, expNumList, 'uni', 0);
        csvDate = arrayfun(@(x) datestr(x, 'yyyy-mm-dd'), csvDataMouse.expDate, 'uni', 0);
        csvRef = cellfun(@(x,y) [x,num2str(y)], csvDate, num2cell(csvDataMouse.expNum), 'uni', 0);
        newExps = ~contains(expRef, csvRef);
    else
        newExps = ones(sum(currIdx),1);
    end
    
    
    if ~any(newExps)
        fprintf('No new data for %s. Skipping... \n', currSub);
        continue; 
    end
    
    dateList = dateList(newExps>0);
    expNumList = expNumList(newExps>0);
    blockPath = blockPath(newExps>0)';
    
    csvBackup(csvPathMouse)
    for i = 1:length(blockPath)
        if ~exist(blockPath{i}, 'file')
            fprintf('No block file for %s %s %s. Skipping... \n', currSub, dateList{i},expNumList{i});
            pause(0.01);
            continue
        end
        
        blk = load(blockPath{i}); blk = blk.block;
        if ~contains(blk.rigName, 'zelda'); continue; end
        newDat.expDate = [newDat.expDate; dateList{i}];
        newDat.expNum = [newDat.expNum; str2double(expNumList{i})];
        
        [~, currExpDef] = fileparts(blk.expDef);
        newDat.expDef = [newDat.expDef; currExpDef];
        newDat.expDuration = [newDat.expDuration; blk.duration];
                newDat.rigName = [newDat.rigName; blk.rigName];

        %%
        fileContents = dir(fileparts(blockPath{i}));
        newDat.sideCam = [newDat.sideCam; max([fileContents(contains({fileContents.name}','sideCam.mj2')).bytes 0])];
        newDat.frontCam = [newDat.frontCam; max([fileContents(contains({fileContents.name}','frontCam.mj2')).bytes 0])];
        newDat.eyeCam = [newDat.eyeCam; max([fileContents(contains({fileContents.name}','eyeCam.mj2')).bytes 0])];
        newDat.micDat = [newDat.micDat; max([fileContents(contains({fileContents.name}','mic.mat')).bytes 0])];
        newDat.path = [newDat.path; {fileparts(blockPath{i})}]; 
        
        ephysPath = fileparts(fileparts(blockPath{i}));
        newDat.ephys = [newDat.ephys; ~isempty(dir([ephysPath filesep '**' filesep '*imec*.ap.bin']))];
    end
    
    if isempty(newDat.expDate)
        fprintf('No new data for %s. Skipping... \n', currSub);
        continue; 
    end
    
    csvDataMouse = [csvDataMouse; struct2table(newDat)]; %#ok<AGROW>
    csvDataMouse = sortrows(csvDataMouse, 'expNum', 'ascend');
    csvDataMouse = sortrows(csvDataMouse, 'expDate', 'ascend');
    try
        csvWriteClean(csvDataMouse, csvPathMouse);
    catch
        fprintf('Issue writing new exps for %s. May be in use. Skipping... \n', currSub);
    end
end
end



function csvBackup(csvLocation)
timeStamp = [datestr(now, 'YYMMDD') '_' datestr(now, 'HHMM')];
[folderLoc, fName] = fileparts(csvLocation);
csvBackupLoc = [folderLoc '\Backups\' timeStamp '_' fName '.csv'];
csvData = readtable(csvLocation);
csvWriteClean(csvData, csvBackupLoc)
end

function csvWriteClean(csvData, csvLocation)
writetable(csvData,csvLocation,'Delimiter',',');
fid = fopen(csvLocation,'rt');
backupDat = fread(fid);
fclose(fid);
backupDat = char(backupDat.');

% replace string S1 with string S2
replaceData = strrep(backupDat, 'NaN', '') ;
replaceData = strrep(replaceData, 'NaT', '') ;
fid = fopen(csvLocation,'wt') ;
fwrite(fid,replaceData) ;
fclose (fid) ;
end