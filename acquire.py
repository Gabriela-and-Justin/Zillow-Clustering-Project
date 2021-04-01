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
          SELECT prop.*, 
                pred.logerror, 
                pred.transactiondate, 
                air.airconditioningdesc, 
                arch.architecturalstyledesc, 
                build.buildingclassdesc, 
                heat.heatingorsystemdesc, 
                landuse.propertylandusedesc, 
                story.storydesc, 
                construct.typeconstructiondesc 

            FROM   properties_2017 prop  
            INNER JOIN (SELECT parcelid,
       					  logerror,
                          Max(transactiondate) transactiondate 
                        FROM   predictions_2017 
                        GROUP  BY parcelid, logerror) pred
                    USING (parcelid) 
                LEFT JOIN airconditioningtype air USING (airconditioningtypeid) 
                LEFT JOIN architecturalstyletype arch USING (architecturalstyletypeid) 
                LEFT JOIN buildingclasstype build USING (buildingclasstypeid) 
                LEFT JOIN heatingorsystemtype heat USING (heatingorsystemtypeid) 
                LEFT JOIN propertylandusetype landuse USING (propertylandusetypeid) 
                LEFT JOIN storytype story USING (storytypeid) 
                LEFT JOIN typeconstructiontype construct USING (typeconstructiontypeid) 
                    WHERE  prop.latitude IS NOT NULL 
                        AND prop.longitude IS NOT NULL
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