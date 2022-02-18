import pandas as pd
import numpy as np
import scipy.io
import smtplib
import datetime
import dateutil.parser

def send_email(mname):    
    server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
    server.login('pinkAVrigs@gmail.com', 'xpr!mnt1')

    receivers = ['takacsflora@gmail.com']
    message = """From: AVrigs <pinkAVrigs@gmail.com>
Subject: Mouse training completed

Hello, 
The following mice have completed their training: 
{}
""".format(mname)


    server.sendmail('pinkAVrigs@gmail.com',receivers,
                    message)


basepath=r'\\zserver.cortexlab.net\Code\AVrig'
mouseList=pd.read_csv(r'%s\!MouseList.csv' % basepath)
activeMice=mouseList['Subject'][mouseList['IsActive']==1].values

numExpToCheck=3

readyMice=[]
for mname in activeMice:
    expinfo=pd.read_csv(r'%s\%s.csv' % (basepath,mname))

    # check whether the mouse is trained on the task
    
    sess2check=expinfo[(expinfo['expDef']=='multiSpaceWorld_checker_training') & (expinfo['expDuration']>600)][-numExpToCheck:]

    
    if (sess2check.shape[0]==numExpToCheck):
        
        # check stages 
        stage=np.zeros(numExpToCheck)
        for i in range(numExpToCheck):
            expPath=sess2check['path'].iloc[i]
            expDate=sess2check['expDate'].iloc[i]
            expNum=sess2check['expNum'].iloc[i]

            block=scipy.io.loadmat(r'%s\%s_%s_%s_Block.mat' % (expPath,expDate,expNum,mname),
                                    squeeze_me=True)
            # for now get some random shit 
            #stage[i]=block['block']['events'].item()['randIdxValues'].item()[0]

            stage[i]=block['block']['events'].item()['selected_paramsetValues'].item()['trainingStage'] 
            
        # check whether they were trained recently
        trainedthisweek=np.zeros(numExpToCheck)
        fiveDaysAgo = datetime.datetime.today() - datetime.timedelta(days=5)
        for i,expDate in enumerate(sess2check['expDate']): 
            mydate = dateutil.parser.parse(expDate)
            if mydate>fiveDaysAgo:
                trainedthisweek[i]=1
            
        if ((np.unique(stage)==5).all()) &((trainedthisweek==1).all()):
            readyMice.append(mname)
    
if len(readyMice)>0:
    print('sending email ...')
    send_email('\n'.join(readyMice))
else: 
    print('no mice are fully trained')
    send_email('no mice are fully trained')