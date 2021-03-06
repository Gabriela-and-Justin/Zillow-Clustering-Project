# Zillow Clustering Project
by: Gabriela Tijerina and Justin Sullivan
****

### Project Summary:
The purpose of this project is to develop a model that is able to predict the log error of Zillow's Zestimate® in predicting home values in three counties in California using Zillow data on single unit/single family properties with transaction dates in 2017. 

<b>What is a log error?</b> 
- Log error is defined as ***logerror=log(Zestimate®)−log(SalePrice)***

<b>How accurate are Zillow Zestimates®?</b> 
- According to <b>[FreeStoneProperties](https://www.freestoneproperties.com/blog/truth-zillow-zestimates/#:~:text=Is%20a%20Zillow%20Zestimate%20High,about%20the%20accuracy%20of%20Zestimates.&text=For%20example%2C%20depending%20on%20the,only%2062%25%20of%20the%20time.)</b> ,
"The median error for larger markets is usually around 2% of the sale price of the home. But the problem with Zestimates is that when they are wrong, they can be significantly wrong. For example, depending on the metro area, Zillow might be within 5% of the sale price only 62% of the time."   

**Data Source:** CodeUp MySQL Database 
****

### Goals:
* Identify the drivers for errors in Zestimates® by incorporating clustering methodologies
* Build a model or series of models to better predict log error for single unit properties in Southern California 
**** 

### Deliverables:
* README.md file explaining what the project is, how to reproduce our work, and our notes from project planning
* A final Jupyter Notebook [(Final_Report_Zillow.ipynb)](https://github.com/Gabriela-and-Justin/Zillow-Clustering-Project/blob/master/Final_Report_Zillow.ipynb) for presentation that includes discoveries we made and work we have done related to uncovering what the drivers of the error in the Zestimate are
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


### Data Science Pipeline:

#### 1. Acquire
- Connect to the SQL company database (login credentials required)
- Summarize initial data to determine how data needs to be prepared and cleaned 
- Define functions to:
    - Create a connection url to access the CodeUp's SQL database using personal credentials
    - Acquire Zillow data from MySQL and return as a dataframe
    - Create a .csv file of acquired data 
- All functions to acquire data are included in [acquire.py](https://github.com/Gabriela-and-Justin/Zillow-Clustering-Project/blob/master/acquire.py)

#### 2. Prepare
- Review data and address any missing or erroneous values 
- Define functions to:
    - Clean Zillow data and return as a clean dataframe 
    - Visualize nulls 
    - Handle missing values 
    - Remove columns 
    - Create features 
    - Handle/remove outliers 
- All functions to prepare the data are included in [prepare.py](https://github.com/Gabriela-and-Justin/Zillow-Clustering-Project/blob/master/prepare.py)

#### 3. Explore
- Address questions posed in planning and brainstorming and figure out drivers of log error
- Create visualizations of variables 
- Run statistical tests 
- Summarize key findings and takeaways
- Define functions to:
    - Split the data to explore on the training data set
    - Run univariate, bivariate, and multivariate visualizations for how features interact with each other and the target, logerror
    - Use clustering to further determine features driving log error and engineer new features as discovered
- All functions to explore the data are included in [explore.py](https://github.com/Gabriela-and-Justin/Zillow-Clustering-Project/blob/master/explore.py)

#### 4. Model/Evaluate
- Goal: develop a model that performs better than the baseline
- Develop a baseline model and linear regression model without controlling for counties
- Iterate process for each of the three counties
- Evaluate if the individualized county models performed better than the all counties model
- Summarize performance, interpret, and document results
- Define functions to:
    - Evaluate model using standard techniques: plotting the residuals, computing the evaluation metrics (SSE, RMSE, and/or MSE), comparing to baseline, plotting y by y-hat
- All functions to evaluate models are included in [evaluate.py](https://github.com/Gabriela-and-Justin/Zillow-Clustering-Project/blob/master/evaluate.py)

#### 5. Deliver
- A final Jupyter Notebook [(Final_Report_Zillow.ipynb)](https://github.com/Gabriela-and-Justin/Zillow-Clustering-Project/blob/master/Final_Report_Zillow.ipynb) for presentation that includes discoveries we made and work we have done related to uncovering what the drivers of the error in the Zestimate® are 

### Conclusion:
- Log error is statistically greater for properties in LA County than properties in Ventura/Orange Counties.
- Log error is statistically different for properties with >5 acres than properties with <5 acres.
- Clustering By Acres and Living Square Footage produces 4 clusters that may help with modeling in future iterations
- Modeling to predict logerror needs additional work and analysis. We built 4 linear regression models, all of which only did marginally better than baseline or performed slightly worse than baseline.
- Individual models controlling for Ventura County and Los Angeles County respectively did perform better in comparison to the all county Linear Regression Model. 

### Next Steps: 
- Utilize clustering observations as features for predicting log error
- Utilize other ML algorithms that may perform better than the linear regression model
- Tweak hyperparameters while continuing to control for county

****

### Instructions for Reproducing Project:  
All files are reproducible and available for download and use. You will need login credentials for access to the Zillow company database.

1.  Read and follow this README.md. 

2.  Download the following files to your working directory:  
 - [Final_Report_Zillow.ipynb](https://github.com/Gabriela-and-Justin/Zillow-Clustering-Project/blob/master/Final_Report_Zillow.ipynb)
 - [acquire.py](https://github.com/Gabriela-and-Justin/Zillow-Clustering-Project/blob/master/acquire.py)
 - [prepare.py](https://github.com/Gabriela-and-Justin/Zillow-Clustering-Project/blob/master/prepare.py)
 - [explore.py](https://github.com/Gabriela-and-Justin/Zillow-Clustering-Project/blob/master/explore.py)
 - [evaluate.py](https://github.com/Gabriela-and-Justin/Zillow-Clustering-Project/blob/master/evaluate.py)
  

3. Run our final Jupyter Notebook [(Final_Report_Zillow.ipynb)](https://github.com/Gabriela-and-Justin/Zillow-Clustering-Project/blob/master/Final_Report_Zillow.ipynb) to reproduce our findings and analysis. 