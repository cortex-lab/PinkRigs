import datetime
import pandas as pd

def check_date_selection(date_selection,dateList):
    date_range = []
    if 'last' in date_selection: 
        date_selection = date_selection.split('last')[1]
        date_range.append(datetime.datetime.today() - datetime.timedelta(days=int(date_selection)))
        date_range.append(datetime.datetime.today())
    else:
        if type(date_selection) is not list:
            date_selection=date_selection.split(':')
        
        for d in date_selection:
            date_range.append(datetime.datetime.strptime(d,'%Y-%m-%d'))   
        #if only one element
        if len(date_range) == 1:
            date_range.append(date_range[0])

    selected_dates = []
    for date in dateList:
        exp_date = datetime.datetime.strptime(date,'%Y-%m-%d')

        if type(date_selection) is list: 
            IsGoodDate= True in ([exp_date==date_range[i] for i in range(len(date_range))])
        else: 
            (exp_date >= date_range[0]) & (exp_date <= date_range[1])
        if IsGoodDate:
            selected_dates.append(True)
        else:
            selected_dates.append(False)
    return selected_dates

def queryExp(mice2check='all',days2check='all',expdef2check='all'):
    """ 
    python version to query experiments based on csvs produced on PinkRigs

    Parameters
    ----
    mice2check : str/list
        selected mice. Can be all, active, or specific subject names
    days2check : str/list
        selected dates. If str: Can be all,lastX,date range, or a single date
                        If list: string of selected dates (e.g. ['2022-03-15','2022-03-30'])
    expdef2check : str
        selected expdef or portion of the string of the the expdef name
        
    Returns
    ----
    exp2checkList : pandas DataFrame 
        concatenated csv of requested experiments and its params 
    """


    root = r'\\zserver.cortexlab.net\Code\AVrig'
    mainCSVLoc = r'%s\!MouseList.csv' % root
    mouseList=pd.read_csv(mainCSVLoc)
    # look for selected mice
    if 'active' in mice2check:
        mouse2checkList = mouseList[mouseList.IsActive==1]['Subject']
    elif 'all' in mice2check: 
        mouse2checkList = mouseList.Subject
    else:
        if not isinstance(mice2check,list):
            mice2check = [mice2check]
        mouse2checkList = mouseList[mouseList.Subject.isin(mice2check)]['Subject']
    exp2checkList = []
    for mm in mouse2checkList:
        expList = pd.read_csv(r'%s\%s.csv' % (root,mm))
        expList.expDate=[expDate.replace('_','-').lower() for expDate in expList.expDate.values]
        if 'all' not in expdef2check:
            expList = expList[expList.expDef.str.contains(expdef2check)]
        if 'all' not in days2check: 
            selected_dates = check_date_selection(days2check,expList.expDate)
            expList = expList[selected_dates]
        
        # add mouse name to list
        expList['Subject'] = mm

        exp2checkList.append(expList)

    exp2checkList = pd.concat(exp2checkList)

    return exp2checkList