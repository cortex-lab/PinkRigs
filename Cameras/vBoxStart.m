function cam = vBoxStart(sessionID)

% This is a sample function. Do not use as is.
% Create a copy in your path outside of this repository, rename it to
% vBoxStart.m and edit according to the cameras you have.

% To start cameras run 'myCam = vBoxStart(sessionID)'
% It is a good idea to have the object myCam accessible from your
% workspace (e.g. in order to execute delete(myCam) to clean things up 
% without restarting Matlab)

% Another option is to run a bash script starting all your cameras in
% separate sessions, e.g. the following two line will start two different 
% Matlab instances with some cameras in each one of them 
% (depending on how the vBoxStart is set up):
% matlab -nodesktop -nosplash -r 'myCam = vBoxStart(1);'
% matlab -nodesktop -nosplash -r 'myCam = vBoxStart(2);'

switch sessionID
    case 1
        cam = Connection('eyeCam');
    case 2
        cam = Connection('frontCam');
    case 3
        cam = Connection('sideCam');
    case 4
        % in case you are ready to run both cameras on the same Matlab
        % session
        cam = Connection('bellyCam');
        cam(2) = Connection('bodyCam');
        % here are the handles to the Figures containing the previews, in 
        % case you want to position the two previews sensibly without the 
        % need to drag the windows around every time
%         hFig(1) = ancestor(cam(1).cameraObj.hPreview, 'Figure');
%         hFig(2) = ancestor(cam(1).cameraObj.hPreview, 'Figure');
%         hFig(1).Position = ...
%         hFig(2).Position = ...
end

