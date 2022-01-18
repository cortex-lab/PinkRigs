%% this funtion will need to be run at the end of each experiment/day? and 
% it ought to copy all video data to the server 

%  inputs: mouse name(s), date
%mname='FT025'; 
%date='2021-07-19'; 

function pipeCams2Server(mname,date)
    ops.serversource='\\128.40.224.65\Subjects'; 
    %ops.serversource='\\znas\Subjects'; 
    ops.localsource='D:\LocalExpData'; % the localExpData folder where data is held
    %ops.localsource='D:\data'; 
    ops.serverRoot=sprintf('%s\\%s\\%s',ops.serversource,mname,date); 
    ops.localRoot=sprintf('%s\\%s\\%s',ops.localsource,mname,date); 

    d=dir([ops.localRoot '\**\*Cam.mj2']);  % checks what camera data is there

    % pipe all  data
    for camidx=1:numel(d)
        % for each camera data check all the relevant other datafiles that are
        % there
        mydat=[d(camidx).folder '\' d(camidx).name]; 
        folders=regexp(d(camidx).folder,'\','split');
        expnum=str2double(folders{numel(folders)}); 

        % target folder 
        servertarget=[ops.serverRoot sprintf('\\%.d',expnum)]; 

        % get all other data files and push them one by one
        camname=regexp(d(camidx).name,'_','split');
        camname=char(camname(4));
        camname=camname(1:end-4);

        % check any files that contain info relating to that camera 
        subd=dir([d(camidx).folder sprintf('\\*%s*',camname)]); 
        
        % pipe all of them into the relevant folder on server
        for fileidx=1:numel(subd)
            % data source local
            mydat=[subd(fileidx).folder '\' subd(fileidx).name];
            % need to get expnum out to be able to identify relevant target folder

            % check if file already exists at the target
            targetfile=[servertarget '\' subd(fileidx).name];
            if isfile(targetfile)
                % check whether file size is the same 
                                
                disp(sprintf('%s copied already.', targetfile))
            else
                disp('copying ...');
                copyfile(mydat,servertarget);
            end
        end 

    end 
end




