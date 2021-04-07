# Zillow Clustering Project: 
by: Gabriela Tijerina and Justin Sullivan
****

### Project Summary:
The purpose of this project is to develop a model that is able to predict the log error of Zillow's Zestimate in predicting home values in three counties in California using Zillow data on single unit/single family properties with transaction dates in 2017. Log error is defined as ***logerror=log(Zestimate)−log(SalePrice)***  

**Data Source:** CodeUp MySQL Database 

### Goals:
* Identify the drivers for errors in Zestimates by incorporating clustering methodologies
* Develop a model that is able to predict log error for Los Angeles County, Orange County, and Ventura County
**** 

### Deliverables:
* README.md file explaining what the project is, how to reproduce our work, and our notes from project planning
* A final Jupyter Notebook [(Final_Report_Zillow.ipynb)](https://github.com/Gabriela-and-Justin/Zillow-Clustering-Project/blob/master/Final_Report_Zillow.ipynb) for presention that includes discoveries we made and work we have done related to uncovering what the drivers of the error in the Zestimate are
* Python files that automate the data acquisition [(acquire.py)](https://github.com/Gabriela-and-Justin/Zillow-Clustering-Project/blob/master/acquire.py), preparation [(prepare.py)](https://github.com/Gabriela-and-Justin/Zillow-Clustering-Project/blob/master/prepare.py), exploration [(explore.py)](https://github.com/Gabriela-and-Justin/Zillow-Clustering-Project/blob/master/explore.py), and evaluation [(evaluate.py)](https://github.com/Gabriela-and-Justin/Zillow-Clustering-Project/blob/master/evaluate.py) process. These modules are imported and used in our final notebook. 
**** 

### Data Dictionary

| Features | Description | Data Type |
|---------|-------------|-----------|
| parcelid | Index: Unique identifier for each property | int64 |
| bathroomcnt | Indicates the number of bathrooms a property has and includes fractional bathrooms | float64 |
| bedroomcnt | Indicates the number of bedrooms a property has | float64 |
| buildingqualitytypeid |  Overall assessment of condition of the building from best (lowest) to worst (highest) | float64 |
| calculatedfinishedsquarefeet | Calculated total finished living area of the property | float64 |
| latitude | Latitude of the middle of the parcel | float64 |
| logerror* | The difference of the Zestimate and the actual sale price | float64 |
| longitude | Longitude of the middle of the parcel | float64 |
| rawcensustractandblock | Tax value of the finished living area on the property| float64 |
| regionidcity | City in which the property is located | float64 |
| regionidzip |  Zip code in which the property is located | float64 |
| structuretaxvaluedollarcnt | The assessed value of the built structure on the parcel | float64 |
| landtaxvaluedollarcnt | Tax value of the land area of the parcel | float64 |
| LA | Indicated if property is in LA County | uint8 |
| Orange | Indicated if property is in Orange County | uint8 |
| Ventura | Indicated if property is in Ventura County | uint8 |
| age | The difference between 2017 and year_built| float64 |
| taxrate | Calculated tax rate for the property | float64 |
| acres | Area of lot (lotsizesquarefeet) converted to acres | float64 |
| cola | Indicates whether the property is in the City of Los Angeles | int64 |
| hot_month_sale | Indicates whether or not property was last sold during the "hot months" of real estate sales (May - Aug)| int64 |
| has_heat | Indicates whether or not the property has heating system | int64 |

\* - Indicates the target feature in this Zillow data
***


## Data Science Pipeline Process:

#### Plan
- State project description and goals
- Explore Zillow data using CodeUp's MySQL database 
- Form initial hypotheses and brainstorm ideas

#### 1. Acquire
- Define functions to:
    - Create a connection url to access the CodeUp's SQL database using personal credentials
    - Acquire Zillow data from MySQL and return as a dataframe
    - Create a .csv file of acquired data 
- All functions to acquire data are included in [acquire.py](https://github.com/Gabriela-and-Justin/Zillow-Clustering-Project/blob/master/acquire.py)
- Summarize initial data to determine how data needs to be prepared and cleaned 

#### 2. Prepare
- Review data and address any missing or erroneous values 
- Define functions to:
    - Clean Zillow data and return as a cleaned pandas DataFrame
- All functions to prepare the data are included in [prepare.py](https://github.com/Gabriela-and-Justin/Zillow-Clustering-Project/blob/master/prepare.py)

#### 3. Explore
- Address questions posed in planning and brainstorming and figure out drivers of log error
- Create visualizations of variables 
- Run statistical tests (correlation and t-test)
- Summarize key findings and takeaways
- Define functions to:
 - Split the dataframe into train, validate, test 
 - Scale the data
- All functions to explore the data are included in [explore.py](https://github.com/Gabriela-and-Justin/Zillow-Clustering-Project/blob/master/explore.py)

#### 4. Model/Evaluate
- Goal: develop a model that performs better than the baseline
- Establish and evaluate a baseline model
- Evaluate model using standard techniques: plotting the residuals, computing the evaluation metrics (SSE, RMSE, and/or MSE), comparing to baseline, plotting y by y-hat
- Choose the best model and test that final model on out-of-sample data
- Summarize performance, interpret, and document results
- All functions to evaluate models are included in [evaluate.py](https://github.com/Gabriela-and-Justin/Zillow-Clustering-Project/blob/master/evaluate.py)

#### 5. Deliver
- A final Jupyter Notebook [Final_Report_Zillow.ipynb](https://github.com/Gabriela-and-Justin/Zillow-Clustering-Project/blob/master/Final_Report_Zillow.ipynb) for presention that includes discoveries we made and work we have done related to uncovering what the drivers of the error in the zestimate are 

****

### Instructions for Reproducing Project: 
- To reproduce project, 