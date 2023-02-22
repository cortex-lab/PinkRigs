classdef optoExpt < handle
    %object which handles interfacing the galvo setup with MC    
    % requires a callback (e.g. optoExpt_callback) to interact with laser. 
    properties
        rig;  
        laser; % DAQ object to operate the laser        
        UDPService;
        expServerObj;
        
        log; % some files to just log things
        filepath;
        AlyxInstance; 
        
    end
    
    methods
        function obj = optoExpt()
            [~, rig] = system('hostname');
            obj.rig = rig(1:end-1);
            UDPListenPort = 1006;     
            
            %Get laser controller object
            obj.laser = LaserController;
            obj.laser.daqSession.Rate = 10000;
            
            % get which hemisphere each patch cord was plugged into
            prompt = {'LED1 hemisphere:','LED2 hemisphere:'};
            answer = inputdlg(prompt);
            obj.laser.hemispheres = answer;
            
            
            %Create basicServices UDP listener to receive alyxInstance info
            %from expServer            
            obj.UDPService = srv.BasicUDPService(obj.rig);
            obj.UDPService.ListenPort = UDPListenPort;
            obj.UDPService.StartCallback = @obj.udpCallback;
            obj.UDPService.bind;            
       end
        
        function udpCallback(obj,src,evt)
            response = regexp(src.LastReceivedMessage,...
                '(?<status>[A-Z]{4})(?<body>.*)\*(?<host>\w*)', 'names');
            [expRef, obj.AlyxInstance] = Alyx.parseAlyxInstance(response.body);
            obj.AlyxInstance.Headless = true;
            disp('this worked.');
        end
        
        function registerListener(obj)
            %Connect to expServer, registering a callback function
            s = srv.StimulusControl.create(sprintf('zelda-stim%s',obj.rig(end)));
            s.connect(true);
            anonListen = @(srcObj, eventObj) optoExpt_callback(eventObj, obj);
            addlistener(s, 'ExpUpdate', anonListen);
            obj.expServerObj = s;
        end
        
        function clearListener(obj)
            obj.expServerObj.disconnect;
            obj.expServerObj.delete;
        end
        
        function stop(obj)
            obj.laser.stop;
        end
        
        function appendToLog(obj,ROW)
            if isempty(obj.log)
                obj.log=ROW;
            else
                fields = fieldnames(obj.log);
                for f = 1:length(fields)
                    obj.log.(fields{f}) = [obj.log.(fields{f}); ROW.(fields{f})];
                end
            end
        end
        
        function saveLog(obj)
            log = obj.log;
            save(obj.filepath, '-struct', 'log');
            
            %If alyx instance available, register to database
            if isempty(obj.AlyxInstance)
                return;
            end            
            subsessionURL = obj.AlyxInstance.SessionURL;
            [dataset,filerecord] = obj.AlyxInstance.registerFile(obj.filepath);
        end       
        
        function delete(obj)
            obj.laser.delete;
            try
                obj.expServerObj.delete;
            catch
            end
        end
        
    end
end