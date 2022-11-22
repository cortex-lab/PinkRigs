%% Get all scripts names

pinkRigsRepo = fileparts(which('zeldaStartup.m'));

% Get function paths, names and extension
funcFiles = dir(fullfile(pinkRigsRepo,'**','*'));
funcFiles(cell2mat(cellfun(@(x) ~contains(x,{'.m','.py','.bat'}), {funcFiles.name}, 'uni', 0))) = [];
funcFiles(cell2mat(cellfun(@(x) strcmp(x,'__init__.py'), {funcFiles.name}, 'uni', 0))) = [];

funcSplit = cellfun(@(x) strsplit(x,'.'), {funcFiles.name}, 'uni', 0);
funcNames = cellfun(@(x) x{1}, funcSplit, 'uni', 0);
funcExt = cellfun(@(x) x{2}, funcSplit, 'uni', 0);

% Add packages to function call name
for ff = 1:numel(funcFiles)
    pack = strsplit(funcFiles(ff).folder,'\\+');
    pack = pack(2:end);
    if ~isempty(pack) && strcmp(funcExt{ff},'m')
        pack = cellfun(@(x) [x '.'], pack, 'uni', 0);
        funcNames{ff} = [strcat(pack{:}) funcNames{ff}];
    end
end

%% Check all scripts to see who calls who

% Loop through
connectivityMatrix = zeros(numel(funcFiles),numel(funcFiles));
for ff = 1:numel(funcFiles)
    funcText = fileread(fullfile(funcFiles(ff).folder,funcFiles(ff).name));
    connectivityMatrix(ff,:) = cell2mat(cellfun(@(x) contains(funcText,x), funcNames, 'uni', 0));
end

%% Build graph

% Build graph
nLabels = strcat(funcNames,'.',funcExt);
G = digraph(connectivityMatrix,nLabels,'omitselfloops');
D = distances(G);

%% Display graph

figure;
p = plot(G,'Layout','force','NodeLabel',nLabels, 'Interpreter', 'none');

% Show successors and predecessors
selectNode = {'natim.main.m'};

% pred
% succ = successors(G,selectNode);
succ = nLabels(D(contains(nLabels,selectNode),:)<inf);
GFsucc = rmedge(G,find((~contains(G.Edges.EndNodes(:,1),selectNode) | ~contains(G.Edges.EndNodes(:,2),succ)) & ...
    (~contains(G.Edges.EndNodes(:,1),succ))));
highlight(p,GFsucc,'EdgeColor',[0.9 0.3 0.1],'NodeColor',[0.9 0.3 0.1])

% succ
pred = predecessors(G,selectNode);
GFpred = rmedge(G,find(~contains(G.Edges.EndNodes(:,2),selectNode) | ~contains(G.Edges.EndNodes(:,1),pred)));
highlight(p,GFpred,'EdgeColor',[0.3 0.9 0.1],'NodeColor',[0.3 0.9 0.1])