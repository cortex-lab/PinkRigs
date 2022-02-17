function cleanEmptyFoldersInDirectory(directory2clean)
% Clean up empty folders
folderList = dir([directory2clean '\**\*']);
folderList = folderList(~ismember({folderList(:).name} ,{'.','..'}));

emptyFolders = folderList([folderList(:).isdir] & [folderList(:).bytes]<5);
emptyFolders = arrayfun(@(x) fullfile(x.folder, x.name), emptyFolders, 'uni', 0);
emptyFolders = flipud(unique(emptyFolders));    
cellfun(@rmdir, emptyFolders);
end
