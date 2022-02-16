function servers = getServersList
    %%% This function will output the current list of data servers, by priority order.
    
    servers = { ...
    '\\zinu.cortexlab.net\Subjects\'; ... %1st
    '\\znas.cortexlab.net\Subjects\'; ... %3rd
    '\\zubjects.cortexlab.net\Subjects\'}; %4th