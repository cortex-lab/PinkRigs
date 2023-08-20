function servers = getServersList
    %% Outputs the current list of data servers, by priority order.
    
    servers = { ...
    '\\zaru.cortexlab.net\Subjects\'; ... %1st
    '\\zinu.cortexlab.net\Subjects\'; ... %2nd
    '\\znas.cortexlab.net\Subjects\'; ... %3rd
    %'\\zubjects.cortexlab.net\Subjects\'; ... %4th
    };