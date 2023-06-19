function my_frontcam()
cam = Connection('frontCam');
vidPreviewHandle = ancestor(cam.cameraObj.hPreview, 'figure');
set(vidPreviewHandle, 'position', [1.4   0.4    0.5926    0.2750]);
% desktop = com.mathworks.mde.desk.MLDesktop.getInstance;
% cmdwin = desktop.getClient('Command Window');
% cmdwin.setBounds(1100,600,450, 550);
end 
