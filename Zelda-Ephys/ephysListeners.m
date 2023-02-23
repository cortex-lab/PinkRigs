[~, rig] = system('hostname');
rig = rig(1:end-1);

if strcmp('Zelda-ephys1',rig)
    optoListener;
end

micListener; 