% save in a tsv the parameters of the recordings
server = '\\zaru.cortexlab.net\Subjects';%dat.path; 
subject = 'AV031';
expDate = '2022-12-16'; 
expNum = '3'; 

optoDat.LaserPower_mW = 15; 
optoDat.Hemisphere = 'L';  

csvData = struct2table(optoDat, 'AsArray', 1);
csvLocation = [server '\' subject '\' expDate '\' expNum '\' expDate '_' expNum '_' subject '_optoMetaData.csv'];
writetable(csvData,csvLocation,'Delimiter',',');