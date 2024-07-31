mice = {'mouse1','mouse2','mouse3'}
folder = 'C:\Users\Célian\Downloads\behaviour';

max_t =  7*60/0.030; % 7 first min at 30Hz

for mm = 1:numel(mice)

    x_apollo = readNPY(fullfile(folder,[mice{mm}, '_apollo'],'dlc_x_tracks.npy'));
    y_apollo = readNPY(fullfile(folder,[mice{mm}, '_apollo'],'dlc_y_tracks.npy'));
    figure; 
    plot(x_apollo(1:max_t),y_apollo(1:max_t))
    title('apollo')

    x_patch = readNPY(fullfile(folder,[mice{mm}, '_optic_fiber_only'],'dlc_x_tracks.npy'));
    y_patch = readNPY(fullfile(folder,[mice{mm}, '_optic_fiber_only'],'dlc_y_tracks.npy'));
    figure; 
    plot(x_patch(1:max_t),y_patch(1:max_t))
    title('patch')
end