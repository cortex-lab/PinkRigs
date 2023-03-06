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

    if namekey == 'naive-all': 
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
            ('FT024','2022-10-12',1,'probe0'),
            ('FT024','2022-10-12',2,'probe0'),
            ('FT024','2022-10-12',3,'probe0'),
            ('FT024','2022-10-12',4,'probe0'),
            ('FT024','2022-10-12',5,'probe0'),
            ('FT024','2022-10-12',1,'probe1'),
            ('FT024','2022-10-12',2,'probe1'),
            ('FT024','2022-10-12',3,'probe1'),
            ('FT024','2022-10-12',4,'probe1'),
            ('FT024','2022-10-12',5,'probe1'),
            ('FT038','2021-11-05',1,'probe0'),
            ('FT038','2021-11-08',1,'probe0'), 
            ('FT039','2021-11-17',1,'probe0'),           
            ('FT039','2021-11-18',1,'probe0'), 
            ('FT039','2021-11-22',1,'probe0'), 
            ('FT039','2021-11-23',1,'probe0'), 
            ('FT039','2021-11-23',3,'probe0'), 
      ]

    elif namekey == 'naive-allen':
        column_names = ['subject','expDate','expNum','probe']
        recordings = [
           ('FT008','2021-01-15',5,'probe0'),
           ('FT008','2021-01-15',5,'probe1'), 
            #('FT008','2021-01-16',8,'probe0'),
            #('FT008','2021-01-16',9,'probe0'),
            # ('FT009','2021-01-19',5,'probe0'), # video
            ('FT009','2021-01-20',7,'probe0'), # video 
            ('FT009','2021-01-20',8,'probe0'),
            ('FT010','2021-03-16',7,'probe0'),
            ('FT010','2021-03-16',7,'probe1'),
            ('FT010','2021-03-16',8,'probe0'),
            ('FT010','2021-03-16',8,'probe1'),
            ('FT011','2021-03-23',6,'probe0'), # video 
           # ('FT011','2021-03-24',6,'probe0'),
           # ('FT011','2021-03-24',7,'probe0'),
        ]

    elif namekey == 'naive-chronic':
        column_names = ['subject','expDate','expNum','probe']
        recordings = [
            ('FT022','2021-07-20',1,'probe0'), #
            ('FT019','2021-07-07',2,'probe0'),
            ('FT025','2021-07-19',4,'probe0'),
            ('FT027','2021-09-13',1,'probe0'),
          #  ('AV024','2022-10-12',1,'probe0'),
            ('AV024','2022-10-12',2,'probe0'),
            ('AV024','2022-10-12',3,'probe0'),
            ('AV024','2022-10-12',4,'probe0'),
            ('AV024','2022-10-12',5,'probe0'),
            #('AV024','2022-10-12',1,'probe1'),
            ('AV024','2022-10-12',2,'probe1'),
            ('AV024','2022-10-12',3,'probe1'),
            ('AV024','2022-10-12',4,'probe1'),
            ('AV024','2022-10-12',5,'probe1'),
            ('FT038','2021-11-05',1,'probe0'),
            ('FT038','2021-11-08',1,'probe0'), 
            ('FT039','2021-11-17',1,'probe0'),           
            ('FT039','2021-11-18',1,'probe0'), 
            ('FT039','2021-11-22',1,'probe0'), 
            ('FT039','2021-11-23',1,'probe0'), 
            ('FT039','2021-11-23',3,'probe0'),
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