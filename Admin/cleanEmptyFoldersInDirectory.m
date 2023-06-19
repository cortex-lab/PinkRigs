function cleanEmptyFoldersInDirectory(directory2clean)
%% Clean up empty folders in a directory
%
% Parameters:
% ------------
% directory2clean (required): string
%   The directory to clean: all empty folders in this directory and the
%   sub-directories will be deleted

folderList = dir([directory2clean '\**\*']);
folderList = folderList(~ismember({folderList(:).name} ,{'.','..'}));

for i = 1:length(folderList)
    folderContents = dir([fullfile(folderList(i).folder, folderList(i).name) '\**\*']);
    folderList(i).bytes = sum([folderContents(:).bytes]);
end

emptyFolders = folderList([folderList(:).isdir] & [folderList(:).bytes]<5);
emptyFolders = arrayfun(@(x) fullfile(x.folder, x.name), emptyFolders, 'uni', 0);
emptyFolders = flipud(unique(emptyFolders));    

for i = 1:length(emptyFolders)
    try rmdir(emptyFolders{i}); catch; end
end
