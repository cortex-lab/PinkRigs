import pandas as pd
import numpy as np
import scipy.io
import datetime
from os.path import exists
import dateutil.parser
import smtplib
from pathlib import Path
import re
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

# import PinkRig utils
from csv_queryExp import get_csv_location
email_path = get_csv_location('training_email')

def send_email(email_text):  
    """
    function to send email about training. 

    Parameters: 
    -----------
    email_text: str 
        large string of email text.

    """
    # Get sender and receiver emails.    
    with open(email_path.__str__()) as f:
        sender_email,pwd = f.read().splitlines()
    receivers_email = ['takacsflora@gmail.com','c.bimbard@ucl.ac.uk','george.booth@ucl.ac.uk','tim.sit.18@ucl.ac.uk']

    msg = MIMEMultipart()
    msg['Subject'] = 'Mouse training completed'
    msg['From'] = sender_email
    msg['To'] = ', '.join(receivers_email)

    # Write Message
    message = MIMEText("""Hello, 
    The following mice have been trained recently: 
    {}

    Cheers!
    AVrig bot""".format(email_text))
    msg.attach(message)

    # Add latest figure about behavior.
    dateToday = datetime.datetime.today().strftime( '%d-%m-%Y')
    figurePath = r'C:\Users\Experiment\Documents\BehaviorFigures\Behavior_' + dateToday + '.png'
    if exists(figurePath):
        with open(figurePath, 'rb') as fp:
            img = MIMEImage(fp.read())
            img.add_header('Content-Disposition',  'attachment',filename='Behavior_' + dateToday + '.png')
            msg.attach(img)

    # Send email.
    server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
    server.login(sender_email, pwd)
    server.sendmail(sender_email,receivers_email,msg.as_string())



mouseList = pd.read_csv(get_csv_location('main'))
activeMice = mouseList['Subject'][mouseList['IsActive']==1].values

deltaDays2Check = 7;

# list of strings with the mice with their training stage
readyMice = []
for mname in activeMice:
    csv_name = get_csv_location(mname)
    if exists(csv_name):
        expinfo = pd.read_csv(get_csv_location(mname))

        # check whether the mouse is trained on the task
        sess2check = expinfo[((expinfo['expDef']=='multiSpaceWorld_checker_training')
                               | (expinfo['expDef']=='multiSpaceWorld_checker_training_block') 
                               | (expinfo['expDef']=='multiSpaceSwitchWorld')) 
                               & (expinfo['expDuration']>600)][-1:]

        if (sess2check.shape[0]>0):
            # take the last day for the update
            expPath = sess2check['expFolder'].iloc[0]
            expDate = re.sub('_','-',sess2check['expDate'].iloc[0])
            try: # date formats aren't homogeneous...
                datetime.datetime.strptime(expDate, '%Y-%m-%d')
            except:
                expDate = datetime.datetime.strptime(expDate, '%d/%m/%Y').strftime('%Y-%m-%d') # convert it to proper format
            expNum = sess2check['expNum'].iloc[0]
            print(mname)
            print(expDate)

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
    send_email('no mice are fully trained')