import pandas as pd
import numpy as np
import scipy.io
import smtplib
import datetime
import dateutil.parser

def send_email(mname):    
    server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
    with open(r'\\zserver.cortexlab.net\Code\AVrig\AVrigEmail.txt') as f:
        address,pwd = f.read().splitlines()

    server.login(address, pwd)

    receivers = ['takacsflora@gmail.com','pipcoen@gmail.com ','magdalena.robacha@gmail.com','c.bimbard@ucl.ac.uk']
    message = """From: AVrigs <{}>
Subject: Mouse training completed

Hello, 
The following mice have completed their training: 
{}
""".format(address,mname)


    server.sendmail(address,receivers,
                    message)


basepath = r'\\zserver.cortexlab.net\Code\AVrig'
mouseList = pd.read_csv(r'%s\!MouseList.csv' % basepath)
activeMice = mouseList['Subject'][mouseList['IsActive']==1].values

readyMice = []
for mname in activeMice:
    expinfo = pd.read_csv(r'%s\%s.csv' % (basepath,mname))

    # check whether the mouse is trained on the task
    
    sess2check = expinfo[(expinfo['expDef']=='multiSpaceWorld_checker_training') & (expinfo['expDuration']>600)][-1:]

    
    if (sess2check.shape[0]>0):

        # maybe send email about current training stage

        expPath = sess2check['path'].iloc[0]
        expDate = sess2check['expDate'].iloc[0]
        try: # date formats aren't homogeneous...
            datetime.datetime.strptime(expDate, '%Y-%m-%d')
        except:
            expDate = datetime.datetime.strptime(expDate, '%d/%m/%Y').strftime('%Y-%m-%d') # convert it to proper format
        expNum = sess2check['expNum'].iloc[0]

        block = scipy.io.loadmat(r'%s\%s_%s_%s_Block.mat' % (expPath,expDate,expNum,mname),squeeze_me=True)

        stage = block['block']['events'].item()['selected_paramsetValues'].item()['trainingStage'] 
            
        # check whether they were trained recently
        yesterday = datetime.datetime.today() - datetime.timedelta(days=1)
        for i,expDate in enumerate(sess2check['expDate']): 
            mydate = dateutil.parser.parse(expDate)
            if mydate >= yesterday:
                trainedthisweek=1
            else: 
                trainedthisweek=0
                   
            
        if trainedthisweek==1:
            readyMice.append('%s - Stage %.0d' % (mname,stage))
    
if len(readyMice)>0:
    print('sending email ...')
    send_email('\n'.join(readyMice))
else: 
    print('no mice are fully trained')
    #send_email('no mice are fully trained')