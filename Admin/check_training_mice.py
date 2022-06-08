import pandas as pd
import numpy as np
import scipy.io
import datetime
from os.path import exists
import dateutil.parser
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

def send_email(mname):
    # Get sender and receiver emails.    
    with open(r'\\zserver.cortexlab.net\Code\AVrig\AVrigEmail.txt') as f:
        sender_email,pwd = f.read().splitlines()
    receivers_email = ['takacsflora@gmail.com','pipcoen@gmail.com ','magdalena.robacha@gmail.com','c.bimbard@ucl.ac.uk']

    msg = MIMEMultipart()
    msg['Subject'] = 'Mouse training completed'
    msg['From'] = sender_email
    msg['To'] = ', '.join(receivers_email)

    # Write Message
    message = MIMEText("""Hello, 
    The following mice have been trained recently: 
    {}

    Cheers!
    AVrig bot""".format(mname))
    msg.attach(message)

    # Add latest figure about behavior.
    dateToday = datetime.datetime.today().strftime( '%d-%m-%Y')
    figurePath = r'C:\Users\Experiment\Documents\BehaviorFigures\Behavior_' + dateToday + '.png'
    if exists(figurePath):
        with open(figurePath, 'rb') as fp:
            img = MIMEImage(fp.read())
            img.add_header('Content-Disposition',  'attachment',filename='Behavior_' + dateToday)
            msg.attach(img)

    # Send email.
    server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
    server.login(sender_email, pwd)
    server.sendmail(sender_email,receivers_email,msg.as_string())


basepath = r'\\zserver.cortexlab.net\Code\AVrig'
mouseList = pd.read_csv(r'%s\!MouseList.csv' % basepath)
activeMice = mouseList['Subject'][mouseList['IsActive']==1].values

deltaDays2Check = 7;

readyMice = []
for mname in activeMice:
    expinfo = pd.read_csv(r'%s\%s.csv' % (basepath,mname))

    # check whether the mouse is trained on the task
    sess2check = expinfo[(expinfo['expDef']=='multiSpaceWorld_checker_training') & (expinfo['expDuration']>600)][-1:]

    if (sess2check.shape[0]>0):
        # take the last day for the update
        expPath = sess2check['expFolder'].iloc[0]
        expDate = sess2check['expDate'].iloc[0]
        try: # date formats aren't homogeneous...
            datetime.datetime.strptime(expDate, '%Y-%m-%d')
        except:
            expDate = datetime.datetime.strptime(expDate, '%d/%m/%Y').strftime('%Y-%m-%d') # convert it to proper format
        expNum = sess2check['expNum'].iloc[0]

        # check whether they were trained recently
        previousDays = datetime.datetime.today() - datetime.timedelta(days=deltaDays2Check)
        dateParsed = dateutil.parser.parse(expDate)

        if dateParsed >= previousDays:
            trainedthisweek=1
            block = scipy.io.loadmat(r'%s\%s_%s_%s_Block.mat' % (expPath,expDate,expNum,mname),squeeze_me=True)

            stage = block['block']['events'].item()['selected_paramsetValues'].item()['trainingStage']
            timeout = block['block']['events'].item()['selected_paramsetValues'].item()['responseWindow']      
            wheelMovementProbability=block['block']['events'].item()['selected_paramsetValues'].item()['wheelMovementProbability']

            readyMice.append('%s - Stage %.0d,timeout in %.1f s, wheel yoked in %.0d%% of trials, on day %s' % (mname,stage,timeout,wheelMovementProbability*100,expDate))

        else: 
            trainedthisweek=0        

    
if len(readyMice)>0:
    now = datetime.datetime.today()
    if not now.strftime("%A") in ["Saturday", "Sunday"]:
        print('sending email ...')
        send_email('\n'.join(readyMice))
else: 
    print('no mice are fully trained')
    #send_email('no mice are fully trained')