
% this code was established because some ephys computers run the opto. 
[~, rig] = system('hostname');
rig = rig(1:end-1);

if strcmp('Zelda-ephys1',rig) || strcmp('Zelda-ephys2',rig)
    optoListener;
end

micListener; 