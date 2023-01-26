function my_sidecam()
cam = Connection('sideCam');
vidPreviewHandle = ancestor(cam.cameraObj.hPreview, 'figure');
set(vidPreviewHandle, 'position', [1.4   0.7    0.5926    0.2750]);
end 
