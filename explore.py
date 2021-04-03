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

def is_outlier(x, lower, upper):
    if (lower >= x) or (x >= upper):
        return 'Yes'
    else:
        
        return 'No'

def split(df, target_var):
    '''
    This function takes in the dataframe and target variable name as arguments and then
    splits the dataframe into train (56%), validate (24%), & test (20%)
    It will return a list containing the following dataframes: train (for exploration), 
    X_train, X_validate, X_test, y_train, y_validate, y_test
    '''
    # split df into train_validate (80%) and test (20%)
    train_validate, test = train_test_split(df, test_size=.20, random_state=13)
    # split train_validate into train(70% of 80% = 56%) and validate (30% of 80% = 24%)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=13)

    # create X_train by dropping the target variable 
    X_train = train.drop(columns=[target_var])
    # create y_train by keeping only the target variable.
    y_train = train[[target_var]]

    # create X_validate by dropping the target variable 
    X_validate = validate.drop(columns=[target_var])
    # create y_validate by keeping only the target variable.
    y_validate = validate[[target_var]]

    # create X_test by dropping the target variable 
    X_test = test.drop(columns=[target_var])
    # create y_test by keeping only the target variable.
    y_test = test[[target_var]]

    partitions = [train, X_train, X_validate, X_test, y_train, y_validate, y_test]
    return partitions

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


def explore_univariate(train, cat_vars, quant_vars):
    '''
    explore each individual categorical variably by: 
    taking in a dataframe and a categorical variable and returning
    a frequency table and barplot of the frequencies. 
    
    explore each individual quantitative variable by:
    takes in a dataframe and a quantitative variable and returns
    descriptive stats table, histogram, and boxplot of the distributions. 
    '''
    for var in cat_vars:
        explore_univariate_categorical(train, var)
        print('_________________________________________________________________')
    for col in quant_vars:
        p, descriptive_stats = explore_univariate_quant(train, col)
        plt.show(p)
        print(descriptive_stats)

def explore_univariate_categorical(train, cat_var):
    '''
    takes in a dataframe and a categorical variable and returns
    a frequency table and barplot of the frequencies. 
    '''
    frequency_table = freq_table(train, cat_var)
    plt.figure(figsize=(2,2))
    sns.barplot(x=cat_var, y='Count', data=frequency_table, color='lightseagreen')
    plt.title(cat_var)
    plt.show()
    print(frequency_table)

def explore_univariate_quant(train, quant_var):
    '''
    takes in a dataframe and a quantitative variable and returns
    descriptive stats table, histogram, and boxplot of the distributions. 
    '''
    descriptive_stats = train[quant_var].describe()
    plt.figure(figsize=(8,2))

    p = plt.subplot(1, 2, 1)
    p = plt.hist(train[quant_var], color='lightseagreen')
    p = plt.title(quant_var)

    # second plot: box plot
    p = plt.subplot(1, 2, 2)
    p = plt.boxplot(train[quant_var])
    p = plt.title(quant_var)
    return p, descriptive_stats
    
def freq_table(train, cat_var):
    '''
    for a given categorical variable, compute the frequency count and percent split
    and return a dataframe of those values along with the different classes. 
    '''
    class_labels = list(train[cat_var].unique())

    frequency_table = (
        pd.DataFrame({cat_var: class_labels,
                      'Count': train[cat_var].value_counts(normalize=False), 
                      'Percent': round(train[cat_var].value_counts(normalize=True)*100,2)}
                    )
    )
    return frequency_table

###################### ________________________________________
#### Bivariate


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

def explore_bivariate_categorical(train, categorical_target, continuous_target, binary):
    '''
    takes in binary categorical variable and binned/categorical target variable, 
    returns a crosstab of frequencies
    runs a chi-square test for the proportions
    and creates a barplot, adding a horizontal line of the overall rate of the binary categorical variable. 
    '''
    print(binary, "\n_____________________\n")
    
    ct = pd.crosstab(train[binary], train[categorical_target], margins=True)
    chi2_summary, observed, expected = run_chi2(train, binary, categorical_target)
    mannwhitney = compare_means(train, continuous_target, binary, alt_hyp='two-sided')
    p = plot_cat_by_target(train, categorical_target, binary)
    
    print("\nMann Whitney Test Comparing Means: ", mannwhitney)
    print(chi2_summary)
#     print("\nobserved:\n", ct)
    print("\nexpected:\n", expected)
    plt.show(p)
    print("\n_____________________\n")
    

    
def run_chi2(train, binary, categorical_target):
    observed = pd.crosstab(train[binary], train[categorical_target])
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    chi2_summary = pd.DataFrame({'chi2': [chi2], 'p-value': [p], 
                                 'degrees of freedom': [degf]})
    expected = pd.DataFrame(expected)
    return chi2_summary, observed, expected


def plot_cat_by_target(train, categorical_target, binary):
    p = plt.figure(figsize=(2,2))
    p = sns.barplot(categorical_target, binary, data=train, alpha=.8, color='lightseagreen')
    overall_rate = train[binary].mean()
    p = plt.axhline(overall_rate, ls='--', color='gray')
    return p

    
def compare_means(train, continuous_target, binary, alt_hyp='two-sided'):
    x = train[train[binary]==0][continuous_target]
    y = train[train[binary]==1][continuous_target]
    return stats.mannwhitneyu(x, y, use_continuity=True, alternative=alt_hyp)

###################### ________________________________________
## Bivariate Quant

def explore_bivariate_quant(train, categorical_target, continuous_target, quant):
    '''
    descriptive stats by each target class. 
    compare means across 2 target groups 
    boxenplot of target x quant
    swarmplot of target x quant
    '''
    print(quant, "\n____________________\n")
    descriptive_stats = train.groupby(categorical_target)[quant].describe().T
    spearmans = compare_relationship(train, continuous_target, quant)
    plt.figure(figsize=(4,4))
    boxen = plot_boxen(train, categorical_target, quant)
#     swarm = plot_swarm(train, categorical_target, quant)
    plt.show()
    scatter = plot_scatter(train, categorical_target, continuous_target, quant)
    plt.show()
    print(descriptive_stats, "\n")
    print("\nSpearman's Correlation Test:\n", spearmans)
    print("\n____________________\n")


def compare_relationship(train, continuous_target, quant):
    return stats.spearmanr(train[quant], train[continuous_target], axis=0)

def plot_swarm(train, categorical_target, quant):
    average = train[quant].mean()
    p = sns.swarmplot(data=train, x=categorical_target, y=quant, color='lightgray')
    p = plt.title(quant)
    p = plt.axhline(average, ls='--', color='black')
    return p

def plot_boxen(train, categorical_target, quant):
    average = train[quant].mean()
    p = sns.boxenplot(data=train, x=categorical_target, y=quant, color='lightseagreen')
    p = plt.title(quant)
    p = plt.axhline(average, ls='--', color='black')
    return p

def plot_scatter(train, categorical_target, continuous_target, quant):
    p = sns.scatterplot(x=quant, y=continuous_target, hue=categorical_target, data=train)
    p = plt.title(quant)
    return p

