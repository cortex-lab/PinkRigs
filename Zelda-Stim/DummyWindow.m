classdef DummyWindow < handle
  %HW.PTB.WINDOW A Psychtoolbox Screen implementation of Window
  %   Detailed explanation goes here
  %
  % Part of Rigbox

  % 2012-10 CB created
  
  properties
    % Background colour of the stimulus window.  Can be a scalar luminance
    % value or an RGB vector.
    BackgroundColour = 0
    % Name of DAQ vendor of device used for the sync pulse echo.  E.g. 'ni'
    DaqVendor = 'ni'
    % The device ID of device to output sync echo pulse on
    DaqDev = 1
    % Channel to output sync echo on e.g. 'port0/line0'. Leave empty for
    % don't use the DAQ
    DaqSyncEchoPort = 'port0/line0'
    % Flag indicating whether PsychToolbox window is open.  See 'Screen
    % OpenWindow?'
    IsOpen = true
  end
  
  properties
    ForegroundColour
    % Screen number to open window in. Screen 0 is always the full Windows
    % desktop.  Screens 1 to n are corresponding to windows display monitors
    % 1 to n.  See 'Screen Screens?'
    ScreenNum = 0
  end
  
  properties
    % A handle to the PTB screen window.  -1 when closed.
    PtbHandle = 1
    % When true stimulus frame should be re-drawn at next opportunity
    Invalid = false
    TimeInvalidated  = -1
    AsyncFlipping = false
    AsyncFlipTimeInvalidated = -1
  end
  
  properties (Access = protected)
    % List of textures currently on the graphics device
    TexList
    DaqSession
  end
  
  methods
    % Window constructor
    function obj = DummyWindow()
    end
        
    function open(~)
    end

    function close(~)
    end

    function delete(~)
    end
    
    function asyncFlipBegin(~)
    end
    
    function [time, invalidFrames, validationLag] = asyncFlipEnd(obj)
      time = now;
      validationLag = 0;
      invalidFrames = 0;
    end

    function [time, invalidFrames, validationLag] = flip(~, ~)
      time = now;
      validationLag = 0;
      invalidFrames = 0;
    end

    function clear(~)
    end

    function drawTexture(varargin)
    end

    function fillRect(~, ~, ~)
    end

    function tex = makeTexture(obj, image)
    end

    function [nx, ny] = drawText(~, ~, ~)
      nx = 0;
      ny = 0;
    end

    function deleteTextures(~)
    end
    
    function applyCalibration(~, ~)      
    end
    
    function c = calibration(varargin)
      c = struct;
    end
  end
  
end