#!/usr/bin/env python
# coding: utf-8

# # Activity: Perform feature engineering 

# ## **Introduction**
# 
# 
# As you're learning, data professionals working on modeling projects use featuring engineering to help them determine which attributes in the data can best predict certain measures.
# 
# In this activity, you are working for a firm that provides insights to the National Basketball Association (NBA), a professional North American basketball league. You will help NBA managers and coaches identify which players are most likely to thrive in the high-pressure environment of professional basketball and help the team be successful over time.
# 
# To do this, you will analyze a subset of data that contains information about NBA players and their performance records. You will conduct feature engineering to determine which features will most effectively predict whether a player's NBA career will last at least five years. The insights gained then will be used in the next stage of the project: building the predictive model.
# 

# ## **Step 1: Imports** 
# 

# Start by importing `pandas`.

# In[1]:


# Import pandas.

import pandas as pd
import numpy as np


# The dataset is a .csv file named `nba-players.csv`. It consists of performance records for a subset of NBA players. As shown in this cell, the dataset has been automatically loaded in for you. You do not need to download the .csv file, or provide more code, in order to access the dataset and proceed with this lab. Please continue with this activity by completing the following instructions.

# In[3]:


# RUN THIS CELL TO IMPORT YOUR DATA.

# Save in a variable named `data`.

data = pd.read_csv("nba-players.csv", index_col=0)

data = pd.read_csv("nba-players.csv", index_col=0)


# <details><summary><h4><strong>Hint 1</strong></h4></summary>
# 
# The `read_csv()` function from `pandas` allows you to read in data from a csv file and load it into a DataFrame.
#     
# </details>

# <details><summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Call the `read_csv()`, pass in the name of the csv file as a string, followed by `index_col=0` to use the first column from the csv as the index in the DataFrame.
#     
# </details>

# ## **Step 2: Data exploration** 

# Display the first 10 rows of the data to get a sense of what it entails.

# In[4]:


# Display first 10 rows of data.

print("First 10 rows of the dataset:")
print(data.head(10))


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# There is a function in the `pandas` library that can be called on a DataFrame to display the first n number of rows, where n is a number of your choice. 
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Call the `head()` function and pass in 10.
# </details>

# Display the number of rows and the number of columns to get a sense of how much data is available to you.

# In[5]:


# Display number of rows, number of columns.

print("\nDataset dimensions:")
print(f"Number of rows: {data.shape[0]}")
print(f"Number of columns: {data.shape[1]}")


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# DataFrames in `pandas` have an attribute that can be called to get the number of rows and columns as a tuple.
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# You can call the `shape` attribute.
# </details>

# **Question:** What do you observe about the number of rows and the number of columns in the data?

# The dataset contains 1340 rows (NBA players) and 21 columns. This gives us a substantial amount of data points to analyze player performance metrics and determine which factors contribute to longer NBA careers.

# Now, display all column names to get a sense of the kinds of metadata available about each player. Use the columns property in pandas.
# 

# In[6]:


# Display all column names.

print("\nColumn names:")
print(data.columns.tolist())


# The following table provides a description of the data in each column. This metadata comes from the data source, which is listed in the references section of this lab.
# 
# <center>
# 
# |Column Name|Column Description|
# |:---|:-------|
# |`name`|Name of NBA player|
# |`gp`|Number of games played|
# |`min`|Number of minutes played per game|
# |`pts`|Average number of points per game|
# |`fgm`|Average number of field goals made per game|
# |`fga`|Average number of field goal attempts per game|
# |`fg`|Average percent of field goals made per game|
# |`3p_made`|Average number of three-point field goals made per game|
# |`3pa`|Average number of three-point field goal attempts per game|
# |`3p`|Average percent of three-point field goals made per game|
# |`ftm`|Average number of free throws made per game|
# |`fta`|Average number of free throw attempts per game|
# |`ft`|Average percent of free throws made per game|
# |`oreb`|Average number of offensive rebounds per game|
# |`dreb`|Average number of defensive rebounds per game|
# |`reb`|Average number of rebounds per game|
# |`ast`|Average number of assists per game|
# |`stl`|Average number of steals per game|
# |`blk`|Average number of blocks per game|
# |`tov`|Average number of turnovers per game|
# |`target_5yrs`|1 if career duration >= 5 yrs, 0 otherwise|
# 
# </center>

# Next, display a summary of the data to get additional information about the DataFrame, including the types of data in the columns.

# In[7]:


# Use .info() to display a summary of the DataFrame.

print("\nDataFrame summary:")
print(data.info())


# **Question:** Based on the preceding tables, which columns are numerical and which columns are categorical?

# Most columns are numerical, including game statistics (gp, min, pts, etc.) and performance metrics (fg, 3p, ft percentages). The only categorical column is 'name', which contains players' names. The target column 'target_5yrs' is binary (0 or 1) representing whether a player's career lasted at least 5 years.

# ### Check for missing values

# Now, review the data to determine whether it contains any missing values. Begin by displaying the number of missing values in each column. After that, use isna() to check whether each value in the data is missing. Finally, use sum() to aggregate the number of missing values per column.
# 

# In[8]:


# Display the number of missing values in each column.
# Check whether each value is missing.
#Aggregate the number of missing values per column.

print("\nMissing values per column:")
print(data.isna().sum())


# **Question:** What do you observe about the missing values in the columns? 

# There don't appear to be any missing values in the dataset. This is beneficial as we won't need to handle missing data through imputation or removal of records.

# **Question:** Why is it important to check for missing values?

# Checking for missing values is crucial because they can significantly impact analysis results. Missing values can:
# 
# -Lead to biased or inaccurate models
# -Cause errors in certain algorithms that can't handle missing data
# -Reduce the effective sample size
# -Indicate potential issues with data collection or processing

# ## **Step 3: Statistical tests** 
# 
# 

# Next, use a statistical technique to check the class balance in the data. To understand how balanced the dataset is in terms of class, display the percentage of values that belong to each class in the target column. In this context, class 1 indicates an NBA career duration of at least five years, while class 0 indicates an NBA career duration of less than five years.

# In[9]:


# Display percentage (%) of values for each class (1, 0) represented in the target column of this dataset.

class_distribution = data['target_5yrs'].value_counts(normalize=True) * 100
print("\nClass distribution in target column (%):")
print(class_distribution)


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# In `pandas`, `value_counts(normalize=True)` can be used to calculate the frequency of each distinct value in a specific column of a DataFrame.  
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# After `value_counts(normalize=True)`, multipling by `100` converts the frequencies into percentages (%).
# </details>

# **Question:** What do you observe about the class balance in the target column?

# The target column shows a reasonable class balance, with approximately 54-55% of players having careers lasting at least 5 years (class 1) and 45-46% having shorter careers (class 0). This relatively balanced distribution is helpful for building a predictive model.

# **Question:** Why is it important to check class balance?

# Checking class balance is important because:
# 
# -Highly imbalanced classes can lead to biased models that favor the majority class
# -Classification algorithms perform better with balanced data
# -Imbalanced data may require special handling techniques (oversampling, undersampling, etc.)
# -It helps determine appropriate evaluation metrics (accuracy can be misleading for imbalanced data)

# ## **Step 4: Results and evaluation** 
# 
# 
# Now, perform feature engineering, with the goal of identifying and creating features that will serve as useful predictors for the target variable, `target_5yrs`. 

# ### Feature selection

# The following table contains descriptions of the data in each column:
# 
# <center>
# 
# |Column Name|Column Description|
# |:---|:-------|
# |`name`|Name of NBA player|
# |`gp`|Number of games played|
# |`min`|Number of minutes played|
# |`pts`|Average number of points per game|
# |`fgm`|Average number of field goals made per game|
# |`fga`|Average number of field goal attempts per game|
# |`fg`|Average percent of field goals made per game|
# |`3p_made`|Average number of three-point field goals made per game|
# |`3pa`|Average number of three-point field goal attempts per game|
# |`3p`|Average percent of three-point field goals made per game|
# |`ftm`|Average number of free throws made per game|
# |`fta`|Average number of free throw attempts per game|
# |`ft`|Average percent of free throws made per game|
# |`oreb`|Average number of offensive rebounds per game|
# |`dreb`|Average number of defensive rebounds per game|
# |`reb`|Average number of rebounds per game|
# |`ast`|Average number of assists per game|
# |`stl`|Average number of steals per game|
# |`blk`|Average number of blocks per game|
# |`tov`|Average number of turnovers per game|
# |`target_5yrs`|1 if career duration >= 5 yrs, 0 otherwise|
# 
# </center>

# **Question:** Which columns would you select and avoid selecting as features, and why? Keep in mind the goal is to identify features that will serve as useful predictors for the target variable, `target_5yrs`. 

# I would avoid selecting:
# 
# 'name': This is a categorical identifier that doesn't provide predictive value for career longevity
# Potentially redundant columns: Some statistics might be derivatives of others (e.g., 'reb' is the sum of 'oreb' and 'dreb')

# Next, select the columns you want to proceed with. Make sure to include the target column, `target_5yrs`. Display the first few rows to confirm they are as expected.

# In[10]:


# Select the columns to proceed with and save the DataFrame in new variable `selected_data`.
# Include the target column, `target_5yrs`.

selected_columns = ['gp', 'min', 'pts', 'fgm', 'fga', 'fg', '3p_made', '3pa', '3p', 
                    'ftm', 'fta', 'ft', 'oreb', 'dreb', 'reb', 'ast', 'stl', 'blk', 
                    'tov', 'target_5yrs']

selected_data = data[selected_columns]


# Display the first few rows.

print("\nFirst few rows of selected data:")
print(selected_data.head())


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Refer to the materials about feature selection and selecting a subset of a DataFrame.
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Use two pairs of square brackets, and place the names of the columns you want to select inside the innermost brackets. 
# 
# </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# There is a function in `pandas` that can be used to display the first few rows of a DataFrame. Make sure to specify the column names with spelling that matches what's in the data. Use quotes to represent each column name as a string. 
# </details>

# ### Feature transformation

# An important aspect of feature transformation is feature encoding. If there are categorical columns that you would want to use as features, those columns should be transformed to be numerical. This technique is also known as feature encoding.

# **Question:** Why is feature transformation important to consider? Are there any transformations necessary for the features you want to use?

# Feature transformation is important because it can:
# 
# -Improve model performance by making the data more suitable for algorithms
# -Normalize or standardize features with different scales
# -Convert categorical variables into numerical form for algorithms that require numerical inputs
# -Reduce dimensionality and multicollinearity
# 
# For the NBA dataset, most features are already numerical and on similar scales (per-game averages). The percentage features (fg, 3p, ft) are already normalized between 0-100%. No complex transformations appear necessary, though normalizing high-range features like 'gp' might be beneficial for certain algorithms.

# ### Feature extraction

# Display the first few rows containing containing descriptions of the data for reference. The table is as follows:
# 
# <center>
# 
# |Column Name|Column Description|
# |:---|:-------|
# |`name`|Name of NBA player|
# |`gp`|Number of games played|
# |`min`|Number of minutes played per game|
# |`pts`|Average number of points per game|
# |`fgm`|Average number of field goals made per game|
# |`fga`|Average number of field goal attempts per game|
# |`fg`|Average percent of field goals made per game|
# |`3p_made`|Average number of three-point field goals made per game|
# |`3pa`|Average number of three-point field goal attempts per game|
# |`3p`|Average percent of three-point field goals made per game|
# |`ftm`|Average number of free throws made per game|
# |`fta`|Average number of free throw attempts per game|
# |`ft`|Average percent of free throws made per game|
# |`oreb`|Average number of offensive rebounds per game|
# |`dreb`|Average number of defensive rebounds per game|
# |`reb`|Average number of rebounds per game|
# |`ast`|Average number of assists per game|
# |`stl`|Average number of steals per game|
# |`blk`|Average number of blocks per game|
# |`tov`|Average number of turnovers per game|
# |`target_5yrs`|1 if career duration >= 5 yrs, 0 otherwise|
# 
# </center>

# In[11]:


# Display the first few rows of `selected_data` for reference.

print("\nFirst few rows of selected data:")
print(selected_data.head())


# **Question:** Which columns lend themselves to feature extraction?

# Several columns could be combined to create more meaningful features:
# 
# -Scoring statistics (pts, fgm, fga, 3p_made, 3pa, ftm, fta) could be used to create efficiency metrics
# -Defensive statistics (stl, blk) could be combined for an overall defensive impact metric
# -Minutes and games played could be used to create durability metrics
# -Points, rebounds, assists, steals, blocks, and turnovers could create an overall contribution metric
# 
# The columns I selected for extraction were:
# 
# 1.'points_per_minute': A measure of scoring efficiency
# 2.'total_contribution': An overall impact metric combining multiple stat categories

# Extract two features that you think would help predict `target_5yrs`. Then, create a new variable named 'extracted_data' that contains features from 'selected_data', as well as the features being extracted.

# In[13]:


# Extract two features that would help predict target_5yrs.
# Create a new variable named `extracted_data`.

extracted_data = selected_data.copy()
extracted_data['points_per_minute'] = extracted_data['pts'] / extracted_data['min']
extracted_data['total_contribution'] = (extracted_data['pts'] + 
                                        extracted_data['reb'] + 
                                        extracted_data['ast'] + 
                                        extracted_data['stl'] + 
                                        extracted_data['blk'] - 
                                        extracted_data['tov'])

print("\nFirst few rows with extracted features:")
print(extracted_data.head())


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Refer to the materials about feature extraction.
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Use the function `copy()` to make a copy of a DataFrame. To access a specific column from a DataFrame, use a pair of square brackets and place the name of the column as a string inside the brackets.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Use a pair of square brackets to create a new column in a DataFrame. The columns in DataFrames are series objects, which support elementwise operations such as multiplication and division. Be sure the column names referenced in your code match the spelling of what's in the DataFrame.
# </details>

# Now, to prepare for the Naive Bayes model that you will build in a later lab, clean the extracted data and ensure ensure it is concise. Naive Bayes involves an assumption that features are independent of each other given the class. In order to satisfy that criteria, if certain features are aggregated to yield new features, it may be necessary to remove those original features. Therefore, drop the columns that were used to extract new features.
# 
# **Note:** There are other types of models that do not involve independence assumptions, so this would not be required in those instances. In fact, keeping the original features may be beneficial.

# In[14]:


# Remove any columns from `extracted_data` that are no longer needed.

columns_to_drop = ['pts', 'reb', 'ast', 'stl', 'blk', 'tov', 'min']
extracted_data = extracted_data.drop(columns=columns_to_drop)


# Display the first few rows of `extracted_data` to ensure that column drops took place.

print("\nFirst few rows after dropping redundant columns:")
print(extracted_data.head())


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Refer to the materials about feature extraction.
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# There are functions in the `pandas` library that remove specific columns from a DataFrame and that display the first few rows of a DataFrame.
# </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Use the `drop()` function and pass in a list of the names of the columns you want to remove. By default, calling this function will result in a new DataFrame that reflects the changes you made. The original DataFrame is not automatically altered. You can reassign `extracted_data` to the result, in order to update it. 
# 
# Use the `head()` function to display the first few rows of a DataFrame.
# </details>

# Next, export the extracted data as a new .csv file. You will use this in a later lab. 

# In[15]:


# Export the extracted data.

extracted_data.to_csv("nba_extracted_features.csv")
print("\nExtracted data exported to 'nba_extracted_features.csv'")


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# There is a function in the `pandas` library that exports a DataFrame as a .csv file. 
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Use the `to_csv()` function to export the DataFrame as a .csv file. 
# </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Call the `to_csv()` function on `extracted_data`, and pass in the name that you want to give to the resulting .csv file. Specify the file name as a string and in the file name. Make sure to include `.csv` as the file extension. Also, pass in the parameter `index` set to `0`, so that when the export occurs, the row indices from the DataFrame are not treated as an additional column in the resulting file. 
# </details>

# ## **Considerations**
# 

# **What are some key takeaways that you learned during this lab? Consider the process you followed and what tasks were performed during each step, as well as important priorities when training data.**

# During this lab, I learned:
# 
# -The importance of understanding the data before feature engineering
# -How to identify and select relevant features for a specific prediction task
# -Techniques for creating new features by combining existing ones
# -The value of removing redundant features when using algorithms with independence assumptions like Naive Bayes
# -The importance of maintaining a clear understanding of the business objective throughout the process

# **What summary would you provide to stakeholders? Consider key attributes to be shared from the data, as well as upcoming project plans.**

# Summary for NBA stakeholders:
# 
# Our feature engineering analysis of NBA player data has identified key performance metrics that are likely to predict career longevity. We've processed data from 1340 players, creating new metrics that capture scoring efficiency and overall contribution to team success.
# 
# The dataset shows a relatively balanced distribution of players with careers lasting 5+ years (approximately 55%) versus those with shorter careers (45%). We've extracted features like points-per-minute and a composite "total contribution" metric that combines scoring, rebounding, playmaking and defensive statistics.
# 
# These engineered features will serve as the foundation for our predictive model in the next phase of the project. The model will help NBA teams identify prospects with the potential for long-term success in the league, supporting more informed draft and player development decisions.
# 
# Our next steps include building and validating a Naive Bayes model using these features, which we'll present in the upcoming project phase.

# **Congratulations!** You've completed this lab. However, you may not notice a green check mark next to this item on Coursera's platform. Please continue your progress regardless of the check mark. Just click on the "save" icon at the top of this notebook to ensure your work has been logged.
