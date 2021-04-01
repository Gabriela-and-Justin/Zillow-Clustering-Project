import pandas as pd
import numpy as np
import os


import matplotlib.pyplot as plt
import seaborn as sns

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

def get_county_dummies(df):
    #Create dummies for county
    dummy_df = pd.get_dummies(df['county'], drop_first = True)
    df = pd.concat([df, dummy_df], axis=1)
    df = df.drop(columns={'county'})

    return df

def train_validate_test_split(df, target, seed=123):
        '''
        This function takes in a dataframe, the name of the target variable
        (for stratification purposes), and an integer for a setting a seed
        and splits the data into train, validate and test. 
        Test is 20% of the original dataset, validate is .30*.80= 24% of the 
        original dataset, and train is .70*.80= 56% of the original dataset. 
        The function returns, in this order, train, validate and test dataframes. 
        '''
        train_validate, test = train_test_split(df, test_size=0.2, 
                                                random_state=seed, 
                                                stratify=df[target])
        train, validate = train_test_split(train_validate, test_size=0.3, 
                                        random_state=seed,
                                        stratify=train_validate[target])
        return train, validate, test

def get_object_cols(df):
    '''
    This function takes in a dataframe and identifies the columns that are object types
    and returns a list of those column names. 
    '''
    # create a mask of columns whether they are object type or not
    mask = np.array(df.dtypes == "object")

        
    # get a list of the column names that are objects (from the mask)
    object_cols = df.iloc[:, mask].columns.tolist()
    
    return object_cols

def get_numeric_cols(df, object_cols):
    '''
    takes in a dataframe and list of object column names
    and returns a list of all other columns names, the non-objects. 
    '''
    numeric_cols = [col for col in df.columns.values if col not in object_cols]
    
    return numeric_cols

def scale_my_data(train, validate, test):
    #call numeric cols
    numeric_cols = get_numeric_cols(train, get_object_cols(train))

    scaler = StandardScaler()
    scaler.fit(train[numeric_cols])

    X_train_scaled = scaler.transform(train[numeric_cols])
    numeric_cols = get_numeric_cols(validate, get_object_cols(validate))
    X_validate_scaled = scaler.transform(validate[[numeric_cols]])
    numeric_cols = get_numeric_cols(test, get_object_cols(test))
    X_test_scaled = scaler.transform(test[numeric_cols])

    return X_train_scaled, X_validate_scaled, X_test_scaled