function my_topcam()
cam = Connection('topCam');
vidPreviewHandle = ancestor(cam.cameraObj.hPreview, 'figure');
set(vidPreviewHandle, 'position', [0.1   0.7    0.5926    0.2750]);
end 
