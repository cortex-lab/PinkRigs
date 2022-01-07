function cam = my_eyecam()
cam = Connection('eyeCam');
vidPreviewHandle = ancestor(cam.cameraObj.hPreview, 'figure');
set(vidPreviewHandle, 'position', [1.4   0.1    0.5926    0.2750]);
end 
