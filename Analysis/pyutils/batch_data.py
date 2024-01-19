import pandas as pd

def get_data_bunch(namekey): 
    """
    contains long list of data that is batch called for certain types of analyses
    Many times data can miss somethings (anatomy/videos etc.)
    There can be separate classes for those cases 

    Parameters:
    -----------
    namekey: str
        identifies the data called. Options: 

    Returns:
    --------
        : pd.DataFrame

    """
    
    if namekey == 'naive-total': 
        column_names = ['subject','expDate','expNum','probe']
        recordings = [
            ('FT008','2021-01-15',5,'probe0'),
            ('FT008','2021-01-15',5,'probe1'), 
            ('FT008','2021-01-16',8,'probe0'),
            ('FT008','2021-01-16',9,'probe0'),
            ('FT009','2021-01-19',5,'probe0'), # video
            ('FT009','2021-01-20',7,'probe0'), # video 
            ('FT009','2021-01-20',8,'probe0'),
            ('FT010','2021-03-16',7,'probe0'),
            ('FT010','2021-03-16',7,'probe1'),
            ('FT010','2021-03-16',8,'probe0'),
            ('FT010','2021-03-16',8,'probe1'),
            ('FT011','2021-03-23',6,'probe0'), # video 
            ('FT011','2021-03-24',6,'probe0'),
            ('FT011','2021-03-24',7,'probe0'),
            ('FT022','2021-07-20',1,'probe0'), # size issue 
            ('FT019','2021-07-07',2,'probe0'), # size issue 
            ('FT025','2021-07-19',4,'probe0'), # size issue 
            ('FT027','2021-09-13',1,'probe0'), # size issue 
            ('AV024','2022-10-12',2,'probe0'),
            ('AV024','2022-10-12',3,'probe0'),
            ('AV024','2022-10-12',4,'probe0'),
            ('AV024','2022-10-12',5,'probe0'),
            ('AV024','2022-10-12',1,'probe1'),
            ('AV024','2022-10-12',2,'probe1'),
            ('AV024','2022-10-12',3,'probe1'),
            ('AV024','2022-10-12',4,'probe1'),
            ('AV024','2022-10-12',5,'probe1'),
            # ('FT038','2021-11-08',1,'probe0'), 
            # ('FT039','2021-11-17',1,'probe0'),           
            # ('FT039','2021-11-18',1,'probe0'), 
            # ('FT039','2021-11-22',1,'probe0'), # facecam alignment issue
            # ('FT039','2021-11-23',1,'probe0'), # facecam alignment issue
            # ('FT039','2021-11-23',3,'probe0'), 
        ]


    elif namekey == 'naive-video-set': 
        column_names = ['subject','expDate','expNum']
        recordings = [
            ('FT008','2021-01-15',5),
            ('FT008','2021-01-16',8),
            ('FT008','2021-01-16',9),
            ('FT009','2021-01-19',5),
            ('FT009','2021-01-20',8),
            ('FT010','2021-03-16',7),
            ('FT010','2021-03-16',8),
            ('FT011','2021-03-24',6),
            ('FT011','2021-03-24',7),
            ('FT022','2021-07-20',1), 
            ('FT019','2021-07-07',2),
            ('FT025','2021-07-19',4),
            ('FT027','2021-09-13',1),
            ('AV024','2022-10-12',1),
            ('AV024','2022-10-12',2),
            ('AV024','2022-10-12',3),
            ('AV024','2022-10-12',4),
            ('AV024','2022-10-12',5),
            ('FT038','2021-11-05',1),
            ('FT038','2021-11-08',1), 
            ('FT039','2021-11-17',1),           
            ('FT039','2021-11-18',1), 
            ('FT039','2021-11-22',1), 
            ('FT039','2021-11-23',1), 
            ('FT039','2021-11-23',3), 
      ]

    elif namekey == 'naive-allen':
        column_names = ['subject','expDate','expNum','probe']
        recordings = [
            ('FT008','2021-01-15',5,'probe0'),
            ('FT008','2021-01-15',5,'probe1'), 
            ('FT008','2021-01-16',8,'probe0'),
            ('FT008','2021-01-16',9,'probe0'),
            ('FT009','2021-01-19',5,'probe0'), # video
            ('FT009','2021-01-20',7,'probe0'), # video 
            ('FT009','2021-01-20',8,'probe0'),
            ('FT010','2021-03-16',7,'probe0'),
            ('FT010','2021-03-16',7,'probe1'),
            ('FT010','2021-03-16',8,'probe0'),
            ('FT010','2021-03-16',8,'probe1'),
            ('FT011','2021-03-23',6,'probe0'), # video 
            ('FT011','2021-03-24',6,'probe0'),
            ('FT011','2021-03-24',7,'probe0'),
            ('FT022','2021-07-20',1,'probe0'), #
            ('FT019','2021-07-07',2,'probe0'),
            ('FT025','2021-07-19',4,'probe0'),
            ('FT027','2021-09-13',1,'probe0'),
        ]

    elif namekey == 'trained-passive-cureated':

        # recordings where I looked at the movement and there seemed to be some sessions with less movement plus independent recordings selected/mouse 
        column_names = ['subject','expDate','expNum','probe']
        recordings = [
            ('AV008','2022-03-10',2,'probe0'),
            ('AV008','2022-03-10',2,'probe1'),
            ('AV008','2022-03-13',3,'probe0'),
            ('AV008','2022-03-13',3,'probe1'),
            ('AV014','2022-06-09',3,'probe0'),
            ('AV014','2022-06-17',2,'probe0'),
            ('AV014','2022-06-17',2,'probe1'),
            ('AV014','2022-07-12',3,'probe0'),
            ('AV014','2022-07-12',3,'probe1'),
            ('AV020','2022-07-27',2,'probe0'),
            ('AV025','2022-11-08',2,'probe0'),
            ('AV025','2022-11-08',2,'probe1'),
            ('AV025','2022-11-10',2,'probe0'),
            ('AV025','2022-11-10',2,'probe1'),
            ('AV030','2022-12-06',4,'probe1'),
            ('AV034','2022-12-08',3,'probe0'),
            ('AV034','2022-12-10',2,'probe0'),

        ]



    elif namekey == 'trained-active-curated':

        # recordings where I looked at the movement and there seemed to be some sessions with less movement plus independent recordings selected/mouse 
        column_names = ['subject','expDate','expNum','probe']
        recordings = [
            ('AV025','2022-11-08',1,'probe0'),
            ('AV025','2022-11-08',1,'probe1'),
            ('AV025','2022-11-10',1,'probe0'),
            ('AV025','2022-11-10',1,'probe1'),
            ('AV030','2022-12-07',2,'probe1'), 
            ('AV034','2022-12-09',1,'probe0'),
            ('AV034','2022-12-10',1,'probe0'),

        ]


    elif namekey == 'naive-acute-rfs':
        column_names = ['subject','expDate','expNum','probe']
        recordings = [
            ('FT008','2021-01-15',1,'probe0'),
            ('FT008','2021-01-15',2,'probe0'),
            ('FT008','2021-01-15',3,'probe0'),
            ('FT008','2021-01-15',4,'probe0'),

            ('FT008','2021-01-15',1,'probe1'), 
            ('FT008','2021-01-15',2,'probe1'), 
            ('FT008','2021-01-15',3,'probe1'), 
            ('FT008','2021-01-15',4,'probe1'),

            #('FT008','2021-01-16',8,'probe0'),
            #('FT008','2021-01-16',9,'probe0'),
            # ('FT009','2021-01-19',5,'probe0'), # video
            ('FT009','2021-01-20',3,'probe0'), # video 
            ('FT009','2021-01-20',4,'probe0'),
            ('FT009','2021-01-20',5,'probe0'),
            ('FT009','2021-01-20',6,'probe0'),
            ('FT010','2021-03-16',3,'probe0'),
            ('FT010','2021-03-16',4,'probe0'),
            ('FT010','2021-03-16',5,'probe0'),
            ('FT010','2021-03-16',6,'probe0'),
            ('FT010','2021-03-16',3,'probe1'),
            ('FT010','2021-03-16',4,'probe1'),
            ('FT010','2021-03-16',5,'probe1'),
            ('FT010','2021-03-16',6,'probe1'),

            ('FT011','2021-03-23',2,'probe0'), # video
            ('FT011','2021-03-23',3,'probe0'), 
            ('FT011','2021-03-23',4,'probe0'), 
            ('FT011','2021-03-23',5,'probe0'), 

           # ('FT011','2021-03-24',6,'probe0'),
           # ('FT011','2021-03-24',7,'probe0'),
       #     ('FT022','2021-07-20',2,'probe0'), #
        #    ('FT019','2021-07-07',3,'probe0'),
            ('FT025','2021-07-19',5,'probe0'),
            ('FT027','2021-09-13',2,'probe0'),
            ('FT038','2021-11-04',1,'probe0'),
            ('FT038','2021-11-04',2,'probe0'),
            ('FT038','2021-11-04',5,'probe0'),
            ('FT038','2021-11-04',6,'probe0'),
            ('FT038','2021-11-04',7,'probe0'),
            ('FT038','2021-11-04',8,'probe0'),   
            ('FT039','2021-11-16',1,'probe0'),                     
            ('FT039','2021-11-16',2,'probe0'), 
            ('FT039','2021-11-16',3,'probe0'),             
            ('FT039','2021-11-16',4,'probe0'), 
            ('FT039','2021-11-16',5,'probe0'),             
            ('FT039','2021-11-16',7,'probe0'), 
            ('FT039','2021-11-16',8,'probe0'), 
            ('FT039','2021-11-16',9,'probe0'), 
            ('AV008','2022-10-26',1,'probe0'),           
            ('AV028','2022-10-26',2,'probe0'), 
            ('AV028','2022-10-26',3,'probe0'), 
            ('AV028','2022-10-26',5,'probe0'), 
            ('AV028','2022-10-26',6,'probe0'), 
            ('AV028','2022-10-26',7,'probe0'), 
            ('AV028','2022-10-26',8,'probe0'), 
            ('AV028','2022-10-26',9,'probe0'), 
            ('AV028','2022-10-26',9,'probe0'), 
            ('AV028','2022-10-26',9,'probe0'), 

        ]
    
    elif namekey == 'active-chronic-rfs':
        column_names = ['subject','expDate','expNum','probe']
        recordings = [
           ('FT030','2021-12-01',3,'probe0'),
           ('FT031','2021-12-03',3,'probe0'),
           ('AV005','2022-05-11',2,'probe0'),
           ('AV005','2022-05-11',3,'probe0'),
           ('AV005','2022-05-11',4,'probe0'),
           ('AV005','2022-05-11',5,'probe0'),
           ('AV008','2022-03-09',2,'probe0'),
           ('AV008','2022-03-09',6,'probe0'),
           ('AV008','2022-03-09',7,'probe0'),
           ('AV008','2022-03-09',8,'probe0'),
           ('AV008','2022-03-09',2,'probe1'),
           ('AV008','2022-03-09',6,'probe1'),
           ('AV008','2022-03-09',7,'probe1'),
           ('AV008','2022-03-09',8,'probe1'),
           ('AV014','2022-06-07',4,'probe0'),
           ('AV014','2022-06-06',1,'probe0'),
           ('AV014','2022-06-07',4,'probe1'),
           ('AV014','2022-06-06',1,'probe1'),
           ('AV020','2022-07-26',3,'probe0'),
           ('AV020','2022-07-26',10,'probe0'),
           ('AV020','2022-07-26',12,'probe0'),
           ('AV030', '2022-12-06',1,'probe0'),
           ('AV030', '2022-12-06',5,'probe0'),
           ('AV030', '2022-12-06',6,'probe0'),
           ('AV030', '2022-12-06',7,'probe0'),
           ('AV030', '2022-12-06',1,'probe1'),
           ('AV030', '2022-12-06',5,'probe1'),
           ('AV030', '2022-12-06',6,'probe1'),
           ('AV030', '2022-12-06',7,'probe1'),
           ('AV034', '2022-12-07',1,'probe0'),
           ('AV034', '2022-12-07',4,'probe0'),
           ('AV034', '2022-12-07',5,'probe0'),
           ('AV034', '2022-12-07',6,'probe0'),
            ('AV025', '2022-11-07',1,'probe0'),
            ('AV025', '2022-11-07',5,'probe0'),
            ('AV025', '2022-11-07',6,'probe0'),
            ('AV025', '2022-11-07',7,'probe0'),
            ('AV025', '2022-11-08',3,'probe0'),
            ('AV025', '2022-11-08',4,'probe0'),
            ('AV025', '2022-11-08',5,'probe0'),
            ('AV025', '2022-11-08',6,'probe0'),
            ('AV025', '2022-11-07',1,'probe1'),
            ('AV025', '2022-11-07',5,'probe1'),
            ('AV025', '2022-11-07',6,'probe1'),
            ('AV025', '2022-11-07',7,'probe1'),
            ('AV025', '2022-11-08',3,'probe1'),
            ('AV025', '2022-11-08',4,'probe1'),
            ('AV025', '2022-11-08',5,'probe1'),
            ('AV025', '2022-11-08',6,'probe1')

        ]
    elif namekey == 'AV008':
            column_names = ['subject','expDate','expNum','probe']
            recordings = [
           ('AV008','2022-03-09',2,'probe0'),
           ('AV008','2022-03-09',6,'probe0'),
           ('AV008','2022-03-09',7,'probe0'),
           ('AV008','2022-03-09',8,'probe0'),
           ('AV008','2022-03-09',2,'probe1'),
           ('AV008','2022-03-09',6,'probe1'),
           ('AV008','2022-03-09',7,'probe1'),
           ('AV008','2022-03-09',8,'probe1'),
            ]

    elif namekey == 'AV005':
            column_names = ['subject','expDate','expNum','probe']
            recordings = [
           ('AV005','2022-05-11',2,'probe0'),
           ('AV005','2022-05-11',3,'probe0'),
           ('AV005','2022-05-11',4,'probe0'),
           ('AV005','2022-05-11',5,'probe0')
            ]

    elif namekey == 'AV014':
            column_names = ['subject','expDate','expNum','probe']
            recordings = [
           ('AV014','2022-06-07',4,'probe0'),
           ('AV014','2022-06-06',1,'probe0'),
           ('AV014','2022-06-07',4,'probe1'),
           ('AV014','2022-06-06',1,'probe1')
            ]
       
    elif namekey == 'AV020':
            column_names = ['subject','expDate','expNum','probe']
            recordings = [
           ('AV020','2022-07-26',3,'probe0'),
           ('AV020','2022-07-26',10,'probe0'),
           ('AV020','2022-07-26',12,'probe0')
            ]
       
    
    elif namekey == 'FT030':
            column_names = ['subject','expDate','expNum','probe']
            recordings = [
           ('FT030','2021-12-01',3,'probe0')
            ]
    elif namekey == 'FT031':
            column_names = ['subject','expDate','expNum','probe']
            recordings = [
            ('FT031','2021-12-03',3,'probe0')
            ]

    elif namekey == 'AV030':
            column_names = ['subject','expDate','expNum','probe']
            recordings = [
            ('AV030', '2022-12-06',1,'probe0'),
            ('AV030', '2022-12-06',5,'probe0'),
            ('AV030', '2022-12-06',6,'probe0'),
            ('AV030', '2022-12-06',7,'probe0'),
            ('AV030', '2022-12-06',1,'probe1'),
            ('AV030', '2022-12-06',5,'probe1'),
            ('AV030', '2022-12-06',6,'probe1'),
            ('AV030', '2022-12-06',7,'probe1')
            ]

    elif namekey == 'AV034':
            column_names = ['subject','expDate','expNum','probe']
            recordings = [
           ('AV034', '2022-12-07',1,'probe0'),
           ('AV034', '2022-12-07',4,'probe0'),
           ('AV034', '2022-12-07',5,'probe0'),
           ('AV034', '2022-12-07',6,'probe0')
            ]

    elif namekey == 'AV025':
            column_names = ['subject','expDate','expNum','probe']
            recordings = [
            ('AV025', '2022-11-07',1,'probe0'),
            ('AV025', '2022-11-07',5,'probe0'),
            ('AV025', '2022-11-07',6,'probe0'),
            ('AV025', '2022-11-07',7,'probe0'),
            ('AV025', '2022-11-08',3,'probe0'),
            ('AV025', '2022-11-08',4,'probe0'),
            ('AV025', '2022-11-08',5,'probe0'),
            ('AV025', '2022-11-08',6,'probe0'),
            ('AV025', '2022-11-07',1,'probe1'),
            ('AV025', '2022-11-07',5,'probe1'),
            ('AV025', '2022-11-07',6,'probe1'),
            ('AV025', '2022-11-07',7,'probe1'),
            ('AV025', '2022-11-08',3,'probe1'),
            ('AV025', '2022-11-08',4,'probe1'),
            ('AV025', '2022-11-08',5,'probe1'),
            ('AV025', '2022-11-08',6,'probe1')
            ]


    elif namekey == 'FT038':
        column_names = ['subject','expDate','expNum','probe']
        recordings = [
            ('FT038','2021-11-05',1,'probe0'),
            ('FT038','2021-11-08',1,'probe0'),
        ]

    elif namekey == 'rf-acute':
        column_names = ['subject','expDate','expNum','probe']
        recordings = [
            ('FT038','2021-11-05',1,'probe0'),
            ('FT038','2021-11-08',1,'probe0'),
        ]

    elif namekey == 'rf-chronic':
        column_names = ['subject','expDate','expNum','probe']
        recordings = [
            ('FT038','2021-11-05',1,'probe0'),
            ('FT038','2021-11-08',1,'probe0'),
        ]

    elif namekey == 'naive-3B':
        column_names = ['subject','expDate','expNum','probe']
        recordings = [
            ('FT022','2021-07-20',1,'probe0'), #
            ('FT019','2021-07-07',2,'probe0'),
            ('FT025','2021-07-19',4,'probe0'),
            ('FT027','2021-09-13',1,'probe0'),
        ]

    elif namekey == 'naive-chronic':
        column_names = ['subject','expDate','expNum','probe']
        recordings = [
            # ('FT022','2021-07-20',1,'probe0'), #
            # ('FT019','2021-07-07',2,'probe0'),
            # ('FT025','2021-07-19',4,'probe0'),
            # ('FT027','2021-09-13',1,'probe0'),
            # ('AV024','2022-10-12',1,'probe0'),    #
            # ('AV024','2022-10-12',2,'probe0'),
            # ('AV024','2022-10-12',3,'probe0'),
            # ('AV024','2022-10-12',4,'probe0'),
            # ('AV024','2022-10-12',5,'probe0'),
            # ('AV024','2022-10-12',1,'probe1'),
            # ('AV024','2022-10-12',2,'probe1'),
            # ('AV024','2022-10-12',3,'probe1'),
            # ('AV024','2022-10-12',4,'probe1'),
            # ('AV024','2022-10-12',5,'probe1'),
            # ('FT038','2021-11-05',1,'probe0'),    #
            # ('FT038','2021-11-08',1,'probe0'),    #
            ('FT039','2021-11-17',1,'probe0'),      #     
            ('FT039','2021-11-18',1,'probe0'),  	#
            ('FT039','2021-11-22',1,'probe0'),      #   
            ('FT039','2021-11-23',1,'probe0'),      #
            ('FT039','2021-11-23',3,'probe0'),      #
        ]


    elif namekey == 'postactive': 
        column_names = ['subject','expDate','expDef','probe']
        recordings = [
            ('AV005', '2022-05-27', 'postactive', 'probe0'),
            ('AV005', '2022-05-25', 'postactive', 'probe0'),
            ('AV005', '2022-05-23', 'postactive', 'probe0'),
            ('AV005', '2022-05-13', 'postactive', 'probe0'),
            ('AV005', '2022-05-12', 'postactive', 'probe0'),
            #('AV008', '2022-03-17', 'postactive', 'probe0'),
            ('AV008', '2022-03-14', 'postactive', 'probe0'),
            ('AV008', '2022-03-12', 'postactive', 'probe0'),
            ('AV008', '2022-03-10', 'postactive', 'probe0'),
            ('AV008', '2022-03-09', 'postactive', 'probe0'),
            ('AV008', '2022-04-06', 'postactive', 'probe0'),
            ('AV014', '2022-06-27', 'postactive', 'probe0'),
            ('AV014', '2022-06-23', 'postactive', 'probe0'),
            ('AV014', '2022-06-21', 'postactive', 'probe0'),
            ('AV014', '2022-06-09', 'postactive', 'probe0'),
            ('AV014', '2022-06-13', 'postactive', 'probe0'),
            ('AV014', '2022-07-08', 'postactive', 'probe0'),
            ('FT030', '2021-12-03', 'postactive', 'probe0'),
            ('FT031', '2021-12-04', 'postactive', 'probe0'),
            ('AV008', '2022-03-31', 'postactive', 'probe1'),
            ('AV008', '2022-03-30', 'postactive', 'probe1'),
            ('AV008', '2022-03-14', 'postactive', 'probe1'),
            ('AV008', '2022-03-23', 'postactive', 'probe1'),
            ('AV008', '2022-03-09', 'postactive', 'probe1'),
            ('AV008', '2022-03-20', 'postactive', 'probe1'),
            ('AV008', '2022-04-06', 'postactive', 'probe1'),
            ('AV014', '2022-06-27', 'postactive', 'probe1'),
            ('AV014', '2022-06-20', 'postactive', 'probe1'),
            ('AV014', '2022-06-16', 'postactive', 'probe1'),
            ('AV014', '2022-06-14', 'postactive', 'probe1'),
            ('AV014', '2022-06-07', 'postactive', 'probe1'),
            ('AV014', '2022-07-08', 'postactive', 'probe1')
        ]

    else: 
        recordings = []
        print('this data calling method is not implemented')    

    recordings = pd.DataFrame(recordings,
        columns= column_names
    )

    return recordings