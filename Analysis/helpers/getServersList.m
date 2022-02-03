function servers = getServersList
    %%% This function will output the current list of data servers, by priority order.
    
    servers = { ...
    '\\zinu.cortexlab.net\Subjects\'; ... %1st
    '\\128.40.224.65\Subjects\'; ... %2nd
    '\\znas.cortexlab.net\Subjects\'; ... %3rd
    '\\zubjects.cortexlab.net\Subjects\'}; %4th