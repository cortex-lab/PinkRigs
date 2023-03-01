function sendPinkRigsEmail(emailSubject, emailContents)

if ~exist('emailContents', 'var'); error('Must provide email contents'); end
if ~exist('emailSubject', 'var'); emailSubject = 'PinkRigs: No Subject Provided'; end
fileID = fopen('\\zinu.cortexlab.net\Subjects\PinkRigs\Helpers\AVrigEmail.txt','r');
emailDeets = textscan(fileID,'%s','delimiter', '\n');
fclose(fileID);


rigEmail = emailDeets{1}{1}; % my gmail address
pwd = emailDeets{1}{2};  % my gmail password 
host = 'smtp.gmail.com';
% sendto = ['takacsflora@gmail.com','pipcoen@gmail.com ','c.bimbard@ucl.ac.uk','george.booth@ucl.ac.uk'];
sendTo = ['pipcoen@gmail.com'];
% preferences

setpref('Internet','SMTP_Server', host);
setpref('Internet','E_mail',rigEmail);
setpref('Internet','SMTP_Username',rigEmail);
setpref('Internet','SMTP_Password',pwd);
props = java.lang.System.getProperties;
props.setProperty('mail.smtp.auth','true');
props.setProperty('mail.smtp.socketFactory.class', 'javax.net.ssl.SSLSocketFactory');
props.setProperty('mail.smtp.socketFactory.port','465');

sendmail(rigEmail,emailSubject,emailContents) ;
end


