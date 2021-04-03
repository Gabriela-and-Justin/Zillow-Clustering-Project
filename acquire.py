#imports
import pandas as pd
import numpy as np
import os
from env import host, user, password
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats

# Establish a connection function
def get_connection(db, user=user, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the CodeUp db.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def new_zillow_data():
    '''
    This function reads the Zillow data from the CodeUp db into a df.
    '''
    sql_query = '''
        select *
        from properties_2017
        join (
			select parcelid, max(logerror) as logerror, max(transactiondate) as transactiondate
			from predictions_2017
			group by parcelid
		    ) as pred_17
	    using(parcelid)
        left join airconditioningtype using(airconditioningtypeid)
        left join architecturalstyletype using(architecturalstyletypeid)
        left join buildingclasstype using(buildingclasstypeid)
        left join heatingorsystemtype using(heatingorsystemtypeid)
        left join storytype using(storytypeid)
        left join typeconstructiontype using(typeconstructiontypeid)
        where year(transactiondate) = 2017;
                '''
    df = pd.read_sql(sql_query, get_connection('zillow'))

    return df


# Acquire Data
def get_zillow_data(cached=False):
    '''
    This function reads in zillow data from Codeup database and returns it
    as a .csv file containing a single dataframe. 
    '''
    
    filename = "zillow.csv"
    if cached == False or os.path.isfile(filename) == False:
        df = new_zillow_data()
        df.to_csv(filename)
    else:
        df = pd.read_csv(filename, index_col=0)
      
   
    return df