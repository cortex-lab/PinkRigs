classdef SignalsExpNoVis < exp.SignalsExp
  %EXP.SIGNALSEXPNOVIS Trial-based Signals Experiments
  %   The class defines a framework for running Signals experiment
  %   definition functions without visual stimuli and provides trial event
  %   Signals along with trial conditions that change each trial.
  %
  % Part of Rigbox

  % 2012-11 CB created
  
  methods    
    function useRig(obj, rig)
      obj.Clock = rig.clock;
      obj.RigName = rig.name;
      obj.DaqController = rig.daqController;
      obj.Wheel = rig.mouseInput;
      if isfield(rig, 'lickDetector')
        obj.LickDetector = rig.lickDetector;
      end
      if ~isempty(obj.DaqController.SignalGenerators)
          outputNames = fieldnames(obj.Outputs); % Get list of all outputs specified in expDef function
          for m = 1:length(outputNames)
              id = find(strcmp(outputNames{m},...
                  obj.DaqController.ChannelNames)); % Find matching channel from rig hardware file
              if id % if the output is present, create callback 
                  obj.Listeners = [obj.Listeners
                    obj.Outputs.(outputNames{m}).onValue(@(v)obj.DaqController.command([zeros(size(v,1),id-1) v])) % pad value with zeros in order to output to correct channel
                    obj.Outputs.(outputNames{m}).onValue(@(v)fprintf('delivering output of %.2f\n',v))
                    ];   
              elseif strcmp(outputNames{m}, 'reward') % special case; rewardValve is always first signals generator in list 
                  obj.Listeners = [obj.Listeners
                    obj.Outputs.reward.onValue(@(v)obj.DaqController.command(v))
                    obj.Outputs.reward.onValue(@(v)fprintf('delivering reward of %.2f\n', v))
                    ];   
              end
          end
      end
    end
    
    function data = run(obj, ref)
      % Runs the experiment
      %
      % run(REF) will start the experiment running, first initialising
      % everything, then running the experiment loop until the experiment
      % is complete. REF is a reference to be saved with the block data
      % under the 'expRef' field, and will be used to ascertain the
      % location to save the data into. If REF is an empty, i.e. [], the
      % data won't be saved.
      
      % Ensure experiment ref exists
      if ~isempty(ref) && ~dat.expExists(ref)
        % If in debug mode, throw warning, otherwise throw as error
        % TODO Propogate debug behaviour to exp.Experiment
        id = 'Rigbox:exp:SignalsExp:experimenDoesNotExist';
        msg = 'Experiment ref ''%s'' does not exist';
        iff(obj.Debug, @() warning(id,msg,ref), @() error(id,msg,ref))
      end
      
      %do initialisation
      init(obj);
      
      obj.Data.rigName = obj.RigName;
      obj.Data.expRef = ref; %record the experiment reference
      
      %Trigger the 'experimentInit' event so any handlers will be called
      initInfo = exp.EventInfo('experimentInit', obj.Clock.now, obj);
      fireEvent(obj, initInfo);
      
      %set pending handler to begin the experiment 'PreDelay' secs from now
      start = exp.EventHandler('experimentInit', exp.StartPhase('experiment'));
      
      % Add callback to update Time is necessary
      start.addCallback(...
        @(~,t)iff(obj.Time.Node.CurrValue, [], @()obj.Time.post(t)));
      % Add callback to update expStart
      start.addCallback(@(varargin)obj.Events.expStart.post(ref));
      obj.Pending = dueHandlerInfo(obj, start, initInfo, obj.Clock.now + obj.PreDelay);
      
      try
        % start the experiment loop
        mainLoop(obj);
        
        %post comms notification with event name and time
        if isempty(obj.AlyxInstance) || ~obj.AlyxInstance.IsLoggedIn
          post(obj, 'AlyxRequest', obj.Data.expRef); %request token from client
          pause(0.2) 
        end
        
        %Trigger the 'experimentCleanup' event so any handlers will be called
        cleanupInfo = exp.EventInfo('experimentCleanup', obj.Clock.now, obj);
        fireEvent(obj, cleanupInfo);
        
        %do our cleanup
        cleanup(obj);
        
        %return the data structure that has been built up
        data = obj.Data;
                
        if ~isempty(ref)
          saveData(obj); %save the data
        end
      catch ex
        obj.IsLooping = false;
        %mark that an exception occured in the block data, then save
        obj.Data.endStatus = 'exception';
        obj.Data.exceptionMessage = ex.message;
        obj.cleanup() % TODO Make cleanup more robust to error states
        if ~isempty(ref)
          saveData(obj); %save the data
        end
        %rethrow the exception
        rethrow(obj.addErrorCause(ex))
      end
    end
       
    
    function sendSignalUpdates(obj)
      try
        if obj.NumSignalUpdates > 0
          post(obj, 'signals', obj.SignalUpdates(1:obj.NumSignalUpdates));
        end
      catch ex
        warning(getReport(ex));
      end
      obj.NumSignalUpdates = 0;
    end
    
    function loadVisual(~, ~)
    end
    function ensureWindowReady(~)
    end
  end
 
  methods (Access = protected)
    
    function cleanup(obj)
      % Performs cleanup after experiment completes
      %
      % cleanup() is called when the experiment is run after the experiment
      % loop completes. Subclasses can override to perform their own 
      % cleanup, but must chain a call to this.
      
      stopdatetime = now;
      
      % collate the logs
      %events
      obj.Data.events = logs(obj.Events);
      %params
      parsLog = obj.ParamsLog.Node.CurrValue;
      obj.Data.paramsValues = [parsLog.value];
      obj.Data.paramsTimes = [parsLog.time];
      %inputs
      obj.Data.inputs = logs(obj.Inputs);
      %outputs
      obj.Data.outputs = logs(obj.Outputs);
      %audio
%       obj.Data.audio = logs(audio);
      
      % MATLAB time stamp for ending the experiment
      obj.Data.endDateTime = stopdatetime;
      obj.Data.endDateTimeStr = datestr(obj.Data.endDateTime);
      
      % some useful data
      obj.Data.duration = etime(...
        datevec(obj.Data.endDateTime), datevec(obj.Data.startDateTime));
       
      % release resources
      obj.Listeners = [];
      KbQueueStop();
      KbQueueRelease();
      
      % delete cached experiment definition function from memory
      [~, exp_func] = fileparts(obj.Data.expDef);
      clear(exp_func)
    end
 
  end
  
end