import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import env


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def visualize_nulls(df):
    '''
    This function takes in a dataframe and returns
    a horizontal bar chart where each bar is a column in the dataframe
    and the bar represents the number of null values in that column
    '''
    missing_df = df.isnull().sum(axis=0).reset_index()
    missing_df.columns = ['column_name', 'missing_count']
    missing_df = missing_df[missing_df['missing_count']>0]
    missing_df = missing_df.sort_values(by='missing_count')
    
    ind = np.arange(missing_df.shape[0])
    width = 0.9
    fig, ax = plt.subplots(figsize=(12,18))
    rects = ax.barh(ind, missing_df.missing_count.values, color='purple')
    ax.set_yticks(ind)
    ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')
    ax.set_xlabel("Count of missing values")
    ax.set_title("Number of missing values in each column")
    plt.show()

def outlier_label(x, lower, upper):
    if (lower >= x) or (x >= upper):
        return 'Yes'
    else:
        
        return 'No'

def outlier_report(df):
    df = wrangle_zillow()
    
    columns_to_check = ['bathroomcnt', 'bedroomcnt',
       'calculatedfinishedsquarefeet',
       'acres', 'rawcensustractandblock','age',
       'structuretaxvaluedollarcnt', 'taxrate',
       'landtaxvaluedollarcnt', 'logerror']
    
    for col in columns_to_check:
        quartile_01, quartile_03 = np.percentile(df[col], [25, 75])
        iqr = quartile_03 - quartile_01
    
        lower_bound = quartile_01 -(3 * iqr)
        upper_bound = quartile_03 +(3 * iqr)
    
        df['outlier'] = df[col].apply(lambda x: outlier_label(x, lower_bound, upper_bound))
    

        print(f"The lower and upper bound of the range for '{col}' respectively is: {lower_bound} and {upper_bound}")

def visualize_outliers(df):
    '''
    This function takes in a dataframe
    and outputs boxplots to show outlier distribution
    for each column in the the dataframe
    '''
    df = wrangle_zillow()
    for col in df.columns:
        sns.boxplot(df[col])
        plt.title(col)
        plt.show()

def get_latitude(df):
    '''
    This function takes in a datafame with latitude formatted as a float,
    converts it to a int and utilizes lambda to return the latitude values
    in a correct format.
    '''
    df.latitude = df.latitude.astype(int)
    df['latitude'] = df['latitude'].apply(lambda x: x / 10 ** (len((str(x))) - 2))
    return df

def get_longitude(df):
    '''This function takes in a datafame with longitude formatted as a float,
    converts it to a int and utilizes lambda to return the longitude values
    in the correct format.
    '''
    df.longitude = df.longitude.astype(int)
    df['longitude'] = df['longitude'].apply(lambda x: x / 10 ** (len((str(x))) - 4))
    return df


def handle_missing_values(df, prop_required_column = .5, prop_required_row = .70):
	#function that will drop rows or columns based on the percent of values that are missing:\
	#handle_missing_values(df, prop_required_column, prop_required_row
    threshold = int(round(prop_required_column*len(df.index),0))
    df.dropna(axis=1, thresh=threshold, inplace=True)
    threshold = int(round(prop_required_row*len(df.columns),0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    return df

def get_counties(df):
    '''
    This function will create dummy variables out of the original fips column. 
    And return a dataframe with all of the original columns except regionidcounty.
    We will keep fips column for data validation after making changes. 
    New columns added will be 'LA', 'Orange', and 'Ventura' which are boolean 
    The fips ids are renamed to be the name of the county each represents. 
    '''
    # create dummy vars of fips id
    county_df = pd.get_dummies(df.fips)
    # rename columns by actual county name
    county_df.columns = ['LA', 'Orange', 'Ventura']
    # concatenate the dataframe with the 3 county columns to the original dataframe
    df_dummies = pd.concat([df, county_df], axis = 1)
    # drop regionidcounty and fips columns
    df_dummies = df_dummies.drop(columns = ['regionidcounty'])
    return df_dummies

def remove_columns(df, cols_to_remove):  
	#remove columns not needed
    df = df.drop(columns=cols_to_remove)
    return df

def create_features(df):
    ''' This function creates feature designed in feature engineering and
    reduces noise but combining certain features into new features
    '''

    #Create feature that is the calculated age of the property
    df['age'] = 2017 - df.yearbuilt

    #Create taxrate variable feature
    df['taxrate'] = df.taxamount/df.taxvaluedollarcnt*100

    #Create acres variable
    df['acres'] = df.lotsizesquarefeet/43560

    #Create feature for whether the property is in the City of Los Angeles
    df['cola'] = df['regionidcity'].apply(lambda x: 1 if x == 12447.0 else 0)

    #Create fetaure for whether or not the prop was last sold during the 'hot months' of real estate sales (May - Aug) 
    df['hot_month_sale'] = df.month_sold.apply(lambda x: 1 if x == 5 or x == 6 or x == 7 or x == 8 else 0)

    #Create Feature for if the property has heating system or not
    df['has_heat'] = df.heatingorsystemdesc.apply(lambda x: 1 if x != 'None' else 0)

    return df

def expand_transactiondate(df):
    #Convert transactiondate to date_time format
    df['transactiondate']=pd.to_datetime(df['transactiondate'], format='%Y-%m-%d')

    #Extract Month for Transaction
    df['month_sold'] = pd.DatetimeIndex(df['transactiondate']).month

    return df


def wrangle_zillow():
    df = pd.read_csv('zillow.csv')
    
    # Restrict df to only properties that meet single use criteria
    single_use = [261, 262, 263, 264, 266, 268, 273, 276, 279]
    df = df[df.propertylandusetypeid.isin(single_use)]
    
    # Restrict df to only those properties with at least 1 bath & bed and 350 sqft area
    df = df[(df.bedroomcnt > 0) & (df.bathroomcnt > 0) & ((df.unitcnt<=1)|df.unitcnt.isnull())\
            & (df.calculatedfinishedsquarefeet>350)]

    # Handle missing values i.e. drop columns and rows based on a threshold
    df = handle_missing_values(df)
    
    # Get Dummies for Counties
    df = get_counties(df)

    # Reset indext to parcel_id
    df.set_index('parcelid', inplace=True)
    
    # drop unnecessary columns
    df = remove_columns(df, ['id',
       'calculatedbathnbr', 'finishedsquarefeet12', 'fullbathcnt', 'heatingorsystemtypeid'
       ,'propertycountylandusecode', 'propertylandusetypeid', 
        'censustractandblock','assessmentyear', 'unitcnt', 'Unnamed: 0', 'propertyzoningdesc'])
    
    # assume that since this is Southern CA, null means 'None' for heating system
    df.heatingorsystemdesc.fillna('None', inplace = True)
    
    # replace nulls with median values for select columns
    df.lotsizesquarefeet.fillna(7313, inplace = True)
    df.buildingqualitytypeid.fillna(6.0, inplace = True)
    
    # format latitude correctly using get_latitude fx
    df = get_latitude(df)

    # format longitude correctly using get_longitudefx
    df = get_longitude(df)

    #Expand Transactiondate
    df = expand_transactiondate(df)

    #Add in new features created
    df = create_features(df)

    # Just to be sure we caught all nulls, drop them here
    df = df.dropna()

    #Drop redudnant columns after creating new features
    df = remove_columns(df, ['heatingorsystemdesc', 'transactiondate', 'month_sold',
                            'yearbuilt', 'taxamount', 'taxvaluedollarcnt','lotsizesquarefeet', 'fips', 'roomcnt'])

    return df

def wrangle_zillow_2nd():
    '''
    This function wrangles zillow in the same ways that the original wrangle_zillow
    does but goes further in a few key aspects based on the exploration of the 
    zillow data:
    - Outliers were discoverd in the following variables and removed:
        - Taxrate: Tax Rates above 5%
        - Acres: Acreage above 12 (~3 STDS)
        - Squarefeet: Properties above 10,000 squareft
    - roomcnt was dropped as a feature...most were zero and this just doesn't make sense
    - Zipcode: 10 wrong zipcodes. For now filling with the most common corresonding zip
    based on lat and long: 91010
    '''
    df = pd.read_csv('zillow.csv')
    
    # Restrict df to only properties that meet single use criteria
    single_use = [261, 262, 263, 264, 266, 268, 273, 276, 279]
    df = df[df.propertylandusetypeid.isin(single_use)]
    
    # Restrict df to only those properties with at least 1 bath & bed and 350 sqft area
    df = df[(df.bedroomcnt > 0) & (df.bathroomcnt > 0) & ((df.unitcnt<=1)|df.unitcnt.isnull())\
            & (df.calculatedfinishedsquarefeet>350)]

    # Handle missing values i.e. drop columns and rows based on a threshold
    df = handle_missing_values(df)
    
    # Get Dummies for Counties
    df = get_counties(df)

    #Replace 10 obersvartions with error in zipcode with corresonding zipcode
    df.regionidzip.replace({399675: 91010}, inplace=True)

    # Reset indext to parcel_id
    df.set_index('parcelid', inplace=True)
    
    # drop unnecessary columns
    df = remove_columns(df, ['id',
       'calculatedbathnbr', 'finishedsquarefeet12', 'fullbathcnt', 'heatingorsystemtypeid'
       ,'propertycountylandusecode', 'propertylandusetypeid', 
        'censustractandblock','assessmentyear', 'unitcnt', 'Unnamed: 0', 'propertyzoningdesc'])
    
    # assume that since this is Southern CA, null means 'None' for heating system
    df.heatingorsystemdesc.fillna('None', inplace = True)
    
    # replace nulls with median values for select columns
    df.lotsizesquarefeet.fillna(7313, inplace = True)
    df.buildingqualitytypeid.fillna(6.0, inplace = True)
    
    # format latitude correctly using get_latitude fx
    df = get_latitude(df)

    # format longitude correctly using get_longitudefx
    df = get_longitude(df)

    #Expand Transactiondate
    df = expand_transactiondate(df)

    #Add in new features created
    df = create_features(df)

    # Just to be sure we caught all nulls, drop them here
    df = df.dropna()

    #Drop redudnant columns after creating new features
    df = remove_columns(df, ['heatingorsystemdesc', 'transactiondate', 'month_sold',
                            'yearbuilt', 'taxamount', 'taxvaluedollarcnt','lotsizesquarefeet', 'fips', 'roomcnt'])

    #Remove Outliers
    df = remove_outliers(df)

    return df

def remove_outliers(df):
    '''
    remove outliers in square feet, acres & tax rate
    '''

    return df[((df.acres < 13) &
               (df.calculatedfinishedsquarefeet < 10000) & 
               (df.taxrate < 6)
              )]