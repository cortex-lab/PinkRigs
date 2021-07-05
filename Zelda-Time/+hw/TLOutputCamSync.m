classdef TLOutputCamSync < hw.TLOutput
 % Copied from TLOutputAcqLive and modified
 % For turning the IR LED for the facecam on/off at the start and end of
 % the experiment to synchronize frames
 
  properties
    DaqDeviceID % The name of the DAQ device ID, e.g. 'Dev1', see DAQ.GETDEVICES
    DaqChannelID % The name of the DAQ channel ID, e.g. 'port1/line0', see DAQ.GETDEVICES
    DaqVendor = 'ni' % Name of the DAQ vendor
    InitialDelay double {mustBeNonnegative} = 0 % sec, time to wait before starting
    PulseDuration {mustBeNonnegative} = Inf; % sec, time that the pulse is on at beginning and end
  end
  
  properties (Transient, Access = private)
    Timer
  end
  
  methods
    function obj = TLOutputCamSync(hw)
      % TLOUTPUTCHRONO Constructor method
      %   Can take the struct form of a previous instance (as saved in the
      %   Timeline hw struct) to intantiate a new object with the same
      %   properties.
      %
      % See Also HW.TIMELINE
      if nargin
        obj.Name = hw.Name;
        obj.DaqDeviceID = hw.DaqDeviceID;
        obj.DaqVendor = hw.DaqVendor;
        obj.DaqChannelID = hw.DaqChannelID;
        obj.InitialDelay = hw.InitialDelay;
        obj.PulseDuration = hw.PulseDuration;
        obj.Enable = hw.Enable;
        obj.Verbose = hw.Verbose;
      else % Some safe defaults
        obj.Name = 'camSync';
        obj.DaqDeviceID = 'Dev1';
        obj.DaqChannelID = 'port0/line2';
      end
    end

    function init(obj, ~)
      % INIT Initialize the output session
      %   INIT(obj, timeline) is called when timeline is initialized.
      %   Creates the DAQ session and ensures it is outputting a low
      %   (digital off) signal.
      %
      % See Also HW.TIMELINE/INIT
      if obj.Enable
        fprintf(1, 'initializing %s\n', obj.toStr);
        obj.Session = daq.createSession(obj.DaqVendor);
        % Turn off warning about clocked sampling availability
        warning('off', 'daq:Session:onDemandOnlyChannelsAdded');
        % Add on-demand digital channel
        obj.Session.addDigitalChannel(obj.DaqDeviceID, obj.DaqChannelID, 'OutputOnly');
        warning('on', 'daq:Session:onDemandOnlyChannelsAdded');
        outputSingleScan(obj.Session, true); % start on
        % If the initial delay is greater than zero, create a timer for
        % starting the signal late
        if obj.InitialDelay > 0
          obj.Timer = timer('StartDelay', obj.InitialDelay);
          obj.Timer.TimerFcn = @(~,~)obj.start();
          obj.Timer.StopFcn = @(src,~)delete(src);
        end
      end
    end
    
    function start(obj, ~) 
      % START Output a high voltage signal
      %   Called when timeline is started, this outputs the first high
      %   voltage signal to triggar external instrument aquisition
      %
      % See Also HW.TIMELINE/START
      if obj.Enable
        % If the initial delay is greater than 0 and the timer is empty,
        % create and start the timer
        if ~isempty(obj.Timer) && obj.InitialDelay > 0 ...
            && strcmp(obj.Timer.Running, 'off')
          start(obj.Timer); % wait for some duration before starting
          return
        end
        
        if obj.Verbose; fprintf(1, 'start %s\n', obj.Name); end
        outputSingleScan(obj.Session, false);
        if obj.PulseDuration ~= Inf
          pause(obj.PulseDuration);
          outputSingleScan(obj.Session, true);
        end
      end
    end
    
    function process(~, ~, ~)
      % PROCESS() Listener for processing acquired Timeline data
      %   PROCESS(obj, source, event) is a listener callback
      %   function for handling tl data acquisition. Called by the
      %   'main' DAQ session with latest chunk of data. 
      %
      % See Also HW.TIMELINE/PROCESS
      %fprintf(1, 'process acqLive\n');
      % -- pass
    end
    
    function stop(obj,~)
        % STOP Stops the DAQ session object.
        %   Called when timeline is stopped.  Outputs a low voltage signal,
        %   the stops and releases the session object.
        %
        % See Also HW.TIMELINE/STOP
        if obj.Enable
          % set digital output OFF
          if obj.PulseDuration ~= Inf
            outputSingleScan(obj.Session, false);
            pause(obj.PulseDuration);
          end
          outputSingleScan(obj.Session, true);
          
          if obj.Verbose; fprintf(1, 'stop %s\n', obj.Name); end
          stop(obj.Session);
          release(obj.Session);
          obj.Session = [];
          obj.Timer = [];
        end
    end
    
    function s = toStr(obj)
      % TOSTR Returns a string that describes the object succintly
      %
      % See Also INIT
        s = sprintf('"%s" on %s/%s (camSync, initial delay %.2f, pulse duration %.2f)',...
            obj.Name, obj.DaqDeviceID, obj.DaqChannelID, obj.InitialDelay, obj.PulseDuration);
    end
  end
  
end