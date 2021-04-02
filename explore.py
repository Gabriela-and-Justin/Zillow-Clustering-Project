import pandas as pd
import numpy as np
import os

# Statistical Tests
import scipy.stats as stats

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

def get_counties():
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

def explore_bivariate(train, categorical_target, continuous_target, binary_vars, quant_vars):
    '''
    This function makes use of explore_bivariate_categorical and explore_bivariate_quant functions. 
    Each of those take in a continuous target and a binned/cut version of the target to have a categorical target. 
    the categorical function takes in a binary independent variable and the quant function takes in a quantitative 
    independent variable. 
    '''
    
    for binary in binary_vars:
        explore_bivariate_categorical(train, categorical_target, continuous_target, binary)
    for quant in quant_vars:
        explore_bivariate_quant(train, categorical_target, continuous_target, quant)

###################### ________________________________________
## Bivariate Categorical

def explore_bivariate_categorical(train, categorical_target, continuous_target, binary_var):
    '''
    takes in binary categorical variable and binned/categorical target variable, 
    returns a crosstab of frequencies
    runs a chi-square test for the proportions
    and creates a barplot, adding a horizontal line of the overall rate of the binary categorical variable. 
    '''
    print(binary_var, "\n_____________________\n")
    ct = pd.crosstab(train[binary_var], train[categorical_target], margins=True)
    chi2_summary, observed, expected = run_chi2(train, binary_var, categorical_target)
    mannwhitney = compare_means(train, continuous_target, binary_var, alt_hyp='two-sided')
    p = plot_cat_by_target(train, categorical_target, binary_var)
    print("\nMann Whitney Test Comparing Means: ", mannwhitney)
    print(chi2_summary)
    print("\nobserved:\n", ct)
    print("\nexpected:\n", expected)
    plt.show(p)
    print("\n_____________________\n")
    

    
def run_chi2(train, binary_var, categorical_target):
    observed = pd.crosstab(train[binary_var], train[categorical_target])
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    chi2_summary = pd.DataFrame({'chi2': [chi2], 'p-value': [p], 
                                 'degrees of freedom': [degf]})
    expected = pd.DataFrame(expected)
    return chi2_summary, observed, expected


def plot_cat_by_target(train, categorical_target, binary_var):
    p = plt.figure(figsize=(2,2))
    p = sns.barplot(categorical_target, binary_var, data=train, alpha=.8, color='lightseagreen')
    #overall_rate = train[binary_var.mean()]
    #p = plt.axhline(overall_rate, ls='--', color='gray')
    return p

def compare_means(train, continuous_target, binary_var, alt_hyp='two-sided'):
    x = train[train[binary_var]==0][continuous_target]
    y = train[train[binary_var]==1][continuous_target]
    return stats.mannwhitneyu(x, y, use_continuity=True, alternative=alt_hyp)

###################### ________________________________________
## Bivariate Quant

def explore_bivariate_quant(train, categorical_target, continuous_target, quant_var):
    '''
    descriptive stats by each target class. 
    boxenplot of target x quant
    swarmplot of target x quant
    Scatterplot
    '''
    print(quant_var, "\n____________________\n")
    descriptive_stats = train.groupby(categorical_target)[quant_var].describe().T
    spearmans = compare_relationship(train, continuous_target, quant_var)
    plt.figure(figsize=(4,4))
    boxen = plot_boxen(train, categorical_target, quant_var)
    #swarm = plot_swarm(train, categorical_target, quant_var)
    plt.show()
    scatter = plot_scatter(train, categorical_target, continuous_target, quant_var)
    plt.show()
    print(descriptive_stats, "\n")
    print("\nSpearman's Correlation Test:\n", spearmans)
    print("\n____________________\n")


def compare_relationship(train, continuous_target, quant_var):
    return stats.spearmanr(train[quant_var], train[continuous_target], axis=0)

#def plot_swarm(train, categorical_target, quant_var):
#    average = train[quant_var].mean()
#    p = sns.swarmplot(data=train, x=categorical_target, y=quant_var, color='lightgray')
 #   p = plt.title(quant_var)
#    p = plt.axhline(average, ls='--', color='black')
 #   return p

def plot_boxen(train, categorical_target, quant_var):
    average = train[quant_var].mean()
    p = sns.boxenplot(data=train, x=categorical_target, y=quant_var, color='lightseagreen')
    p = plt.title(quant_var)
    p = plt.axhline(average, ls='--', color='black')
    return p

def plot_scatter(train, categorical_target, continuous_target, quant_var):
    p = sns.scatterplot(x=quant_var, y=continuous_target, hue=categorical_target, data=train)
    p = plt.title(quant_var)
    return p
