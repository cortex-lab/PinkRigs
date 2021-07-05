function micUDPCallback(src, event)

% fprintf('%s:%d\n', u.DatagramAddress, u.DatagramPort),
% u.RemoteHost=u.DatagramAddress;
% u.RemotePort=u.DatagramPort;
% disp('now reading data');

persistent file2save ar fileOnServer


t = now;
ip=src.DatagramAddress;
port=src.DatagramPort;
data=fread(src);
str=char(data');
tag = ['[micUDP ', datestr(t, 'HH:MM:SS.FFF'), '] '];
fprintf('%sReceived ''%s'' from %s:%d\n', tag, str, ip, port);

info=dat.mpepMessageParse(str);

% update remote IP to that of the sender
src.RemoteHost = ip;
src.RemotePort = port;

switch info.instruction
    case 'hello'
        fwrite(src, data);
    case 'ExpStart'
        % configure save filename and path
        [filePath, fileStem] = dat.expPath(info.expRef, 'main', 'local');
        file2save = fullfile(filePath, [fileStem, '_mic.mat']);
        [filePath, fileStem] = dat.expPath(info.expRef, 'main', 'master');
        fileOnServer = fullfile(filePath, [fileStem, '_mic.mat']);
        adi = audiodevinfo;

        % Find the ID of the correct input
        tmp = cellfun(@strfind, {adi.input.Name}, repmat({'UltraMic'}, size({adi.input.Name})), ...
            'UniformOutput', false);
        iDevice = find(~cellfun(@isempty, tmp));
%         [~, iDevice] = ismember('Microphone (UltraMic 200K 16 bit r4) (Windows DirectSound)', ...
%             {adi.input.Name});
        
        % define the audiorecorder object
        Fs = 192e3;
        nBits = 16;
        nChannels = 1;
        ar = audiorecorder(Fs, nBits, nChannels, adi.input(iDevice).ID);
        
        % start non-blocking asynchronous recording
        record(ar);

        fwrite(src, data);
        
    case {'ExpEnd', 'ExpInterrupt'}
        stop(ar);
        Fs = ar.SampleRate;
        nBits = ar.BitsPerSample;
        micData = getaudiodata(ar, 'int16');
%         figure, plot(micData)
        delete(ar);
        clear ar;
        
        % save data to disk
        [folder, ~, ~] = fileparts(file2save);
        if ~exist(folder, 'dir')
            mkdir(folder);
        end
        fprintf('Saving data to disk..');
        % 2021-05-26: Tim: changing to save 7.3
        save(file2save, 'micData', 'Fs', 'nBits', '-v7.3');
        fprintf('.done\n');
        fprintf('Copying data to server..');
        [status, message] = copyfile(file2save, fileOnServer);
        fprintf('.done\n');
        if ~status
            warning('Copying the server failed with the following message:');
            warning(sprintf('%s', message));
        end
        

        fwrite(src, data);
    case 'BlockStart'
        fwrite(src, data);
    case 'BlockEnd'
        fwrite(src, data);
    case 'StimStart'
        fwrite(src, data);
    case 'StimEnd'
        fwrite(src, data);
    case 'alyx' % Alyx instance recieved
        fwrite(src, data);
    otherwise
        fprintf('Unknown instruction : %s', info.instruction);
        fwrite(src, data);
end


end
%===========================================================
%

