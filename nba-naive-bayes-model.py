#!/usr/bin/env python
# coding: utf-8

# # Activity: Build a Naive Bayes model 

# ## Introduction
# 
# In this activity, you will build your own Naive Bayes model. Naive Bayes models can be valuable to use any time you are doing work with predictions because they give you a way to account for new information. In today's world, where data is constantly evolving, modeling with Naive Bayes can help you adapt quickly and make more accurate predictions about what could occur.
# 
# For this activity, you work for a firm that provides insights for management and coaches in the National Basketball Association (NBA), a professional basketball league in North America. The league is interested in retaining players who can last in the high-pressure environment of professional basketball and help the team be successful over time. In the previous activity, you analyzed a subset of data that contained information about the NBA players and their performance records. You conducted feature engineering to determine which features would most effectively predict a player's career duration. You will now use those insights to build a model that predicts whether a player will have an NBA career lasting five years or more. 
# 
# The data for this activity consists of performance statistics from each player's rookie year. There are 1,341 observations, and each observation in the data represents a different player in the NBA. Your target variable is a Boolean value that indicates whether a given player will last in the league for five years. Since you previously performed feature engineering on this data, it is now ready for modeling.   

# ## Step 1: Imports

# ### Import packages
# 
# Begin with your import statements. Of particular note here are `pandas` and from `sklearn`, `naive_bayes`, `model_selection`, and `metrics`.

# In[1]:


# Import relevant libraries and modules.
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# ### Load the dataset
# 
# Recall that in the lab about feature engineering, you outputted features for the NBA player dataset along with the target variable ``target_5yrs``. Data was imported as a DataFrame called `extracted_data`. As shown in this cell, the dataset has been automatically loaded in for you. You do not need to download the .csv file, or provide more code, in order to access the dataset and proceed with this lab. Please continue with this activity by completing the following instructions.

# In[2]:


# RUN THIS CELL TO IMPORT YOUR DATA.
# Load extracted_nba_players_data.csv into a DataFrame called extracted_data.

extracted_data = pd.read_csv('extracted_nba_players_data.csv')


# ### Display the data
# 
# Review the first 10 rows of data.

# In[3]:


# Display the first 10 rows of data.

# Display the first 10 rows of data
extracted_data.head(10)


# ## Step 2: Model preparation

# ### Isolate your target and predictor variables
# Separately define the target variable (`target_5yrs`) and the features.

# In[4]:


# Define the y (target) variable.

y = extracted_data['target_5yrs']


# Define the X (predictor) variables.

X = extracted_data.drop('target_5yrs', axis=1)


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Refer to [the content about splitting your data into X and y](https://www.coursera.org/learn/the-nuts-and-bolts-of-machine-learning/lecture/VxbUT/construct-a-naive-bayes-model-with-python).
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# In `pandas`, subset your DataFrame by using square brackets `[]` to specify which column(s) to select.
# </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Quickly subset a DataFrame to exclude a particular column by using the `drop()` function and specifying the column to drop.
# </details>

# ### Display the first 10 rows of your target data
# 
# Display the first 10 rows of your target and predictor variables. This will help you get a sense of how the data is structured.

# In[5]:


# Display the first 10 rows of your target data.

y.head(10)


# **Question:** What do you observe about the your target variable?
# 

# The target variable appears to be binary (True/False or 1/0), indicating whether a player's career lasted 5+ years. It's a Boolean variable, which makes this a binary classification problem.

# In[6]:


# Display the first 10 rows of your predictor variables.

X.head(10)


# **Question:** What do you observe about the your predictor variables?

# The predictor variables consist of various numeric features related to NBA players' performance statistics from their rookie year. These features have different scales and represent different aspects of a player's performance that might influence their career duration.

# ### Perform a split operation on your data
# 
# Divide your data into a training set (75% of data) and test set (25% of data). This is an important step in the process, as it allows you to reserve a part of the data that the model has not observed. This tests how well the model generalizes—or performs—on new data.

# In[7]:


# Perform the split operation on your data.
# Assign the outputs as follows: X_train, X_test, y_train, y_test.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)



# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Refer to [the content about splitting your data between a training and test set](https://www.coursera.org/learn/the-nuts-and-bolts-of-machine-learning/lecture/VxbUT/construct-a-naive-bayes-model-with-python).
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Call the function in the `model_selection` module of `sklearn` on the features and target variable, in order to perform the splitting.
# </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Call the `model_selection.train_test_split()` function, passing in both `features` and `target`, while configuring the appropriate `test_size`.
# 
# Assign the output of this split as `X_train`, `X_test`, `y_train`, `y_test`.
# </details>

# ### Print the shape of each output 
# 
# Print the shape of each output from your train-test split. This will verify that the split operated as expected.

# In[8]:


# Print the shape (rows, columns) of the output from the train-test split.

# Print the shape of X_train.

print("Shape of X_train:", X_train.shape)



# Print the shape of X_test.

print("Shape of X_test:", X_test.shape)



# Print the shape of y_train.

print("Shape of y_train:", y_train.shape)



# Print the shape of y_test.

print("Shape of y_test:", y_test.shape)


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Call the attribute that DataFrames in `pandas` have to get the number of rows and number of columns as a tuple.
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Call the `shape` attribute.
# </details>

# **Question:** How many rows are in each of the outputs?
# 

# Based on a dataset with 1,341 observations, the split created:
# 
# -X_train: ~1,005 rows (75% of the data) and the same number of columns as X
# -X_test: ~336 rows (25% of the data) and the same number of columns as X
# -y_train: ~1,005 rows (matching X_train)
# -y_test: ~336 rows (matching X_test)

# **Question:** What was the effect of the train-test split?
# 

# The train-test split divided our data into two portions - a larger portion (75%) for training the model and a smaller portion (25%) for testing the model's performance on unseen data. This helps us evaluate how well the model generalizes to new data.

# ## Step 3: Model building

# **Question:** Which Naive Bayes algorithm should you use?

# We should use GaussianNB (Gaussian Naive Bayes) because our predictor variables appear to be continuous numerical values representing player statistics. GaussianNB assumes that the features follow a normal distribution, which is appropriate for continuous data like basketball performance metrics.

# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Refer to [the content about different implementations of the Naive Bayes](https://www.coursera.org/learn/the-nuts-and-bolts-of-machine-learning/supplement/1zfDy/naive-bayes-classifiers) to determine which is appropriate in this situation.
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Note that you are performing binary classification.
# </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# You can identify the appropriate algorithm to use because you are performing a binary classification and assuming that the features of your model follow a normal distribution.
# </details>

# ### Fit your model to your training data and predict on your test data
# 
# By creating your model, you will be drawing on your feature engineering work by training the classifier on the `X_train` DataFrame. You will use this to predict `target_5yrs` from `y_train`.
# 
# Start by defining `nb` to be the relevant algorithm from `sklearn`.`naive_bayes`. Then fit your model to your training data. Use this fitted model to create predictions for your test data.

# In[9]:


# Assign `nb` to be the appropriate implementation of Naive Bayes.

nb = GaussianNB()



# Fit the model on your training data.

nb.fit(X_train, y_train)



# Apply your model to predict on your test data. Call this "y_pred".

y_pred = nb.predict(X_test)


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Refer to [the content about constructing a Naive Bayes](https://www.coursera.org/learn/the-nuts-and-bolts-of-machine-learning/lecture/VxbUT/construct-a-naive-bayes-model-with-python).
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# The appropriate implementation in this case is `naive_bayes`.`GaussianNB()`. Fit this model to your training data and predict on your test data.
# </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Call `fit()`and pass your training feature set and target variable. Then call `predict()` on your test feature set.
# </details>

# ## Step 4: Results and evaluation
# 

# ### Leverage metrics to evaluate your model's performance
# 
# To evaluate the data yielded from your model, you can leverage a series of metrics and evaluation techniques from scikit-learn by examining the actual observed values in the test set relative to your model's prediction. Specifically, print the accuracy score, precision score, recall score, and f1 score associated with your test data and predicted values.

# In[10]:


# Print your accuracy score.

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {accuracy:.4f}")



# Print your precision score.

precision = precision_score(y_test, y_pred)
print(f"Precision Score: {precision:.4f}")



# Print your recall score.

recall = recall_score(y_test, y_pred)
print(f"Recall Score: {recall:.4f}")



# Print your f1 score.

f1 = f1_score(y_test, y_pred)
print(f"F1 Score: {f1:.4f}")


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Refer to [the content about model evaluation](https://www.coursera.org/learn/the-nuts-and-bolts-of-machine-learning/lecture/EITmV/key-evaluation-metrics-for-classification-models) for detail on these metrics.
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# The `metrics` module in `sklearn` has a function for computing each of these metrics.
# </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Call `accuracy_score()`, `precision_score()`, `recall_score()`, and `f1_score()`, passing `y_test`, and `y_pred` into each function.
# </details>

# **Question:** What is the accuracy score for your model, and what does this tell you about the success of the model's performance?
# 
# 

# The accuracy score of 0.6537 indicates that our model correctly predicted the career duration (over/under 5 years) for about 65.4% of the players in the test set. This is somewhat better than random guessing, but shows that the model has significant room for improvement in its overall predictive capability.

# **Question:** Can you evaluate the success of your model by using the accuracy score exclusively?
# 

# No, we cannot evaluate the success of the model using accuracy alone. The relatively moderate accuracy of 65.4% doesn't tell the complete story. Looking at our other metrics reveals important imbalances in the model's performance that accuracy alone obscures. This is why we need to examine precision, recall, and the F1 score as well.

# **Question:** What are the precision and recall scores for your model, and what do they mean? Is one of these scores more accurate than the other?
# 

# Our model has a high precision score of 0.8382, which means that when it predicts a player will last 5+ years, it's right about 84% of the time. However, the recall score is only 0.5481, meaning the model only identifies about 55% of players who actually do last 5+ years. This indicates that while our model is cautious in making positive predictions (fewer false positives), it misses many players who would actually have longer careers (more false negatives). In this context, precision is significantly higher than recall, suggesting the model is conservative in its predictions of long careers.

# **Question:** What is the F1 score of your model, and what does this score mean?

# The F1 score of 0.6628 represents the harmonic mean of our precision and recall scores. This moderate F1 score reflects the imbalance between our high precision and lower recall. It suggests that while the model is good at avoiding false positives, it needs improvement in detecting more of the true positives. For NBA team management, this means the model is reliable when it predicts long careers but may miss potential long-term talents.

# ### Gain clarity with the confusion matrix
# 
# Recall that a confusion matrix is a graphic that shows your model's true and false positives and negatives. It helps to create a visual representation of the components feeding into the metrics.
# 
# Create a confusion matrix based on your predicted values for the test set.

# In[11]:


# Construct and display your confusion matrix.

# Construct the confusion matrix for your predicted and test values.

cm = confusion_matrix(y_test, y_pred)



# Create the display for your confusion matrix.

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['<5 years', '≥5 years'],
            yticklabels=['<5 years', '≥5 years'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')



# Plot the visual in-line.

plt.show()


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# The `metrics` module has functions to create a confusion matrix.
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Call `confusion_matrix`, passing in `y_test` and `y_pred`. Then, utilize `ConfusionMatrixDisplay()` to display your confusion matrix.
# </details>

# **Question:** What do you notice when observing your confusion matrix, and does this correlate to any of your other calculations?
# 

# The confusion matrix shows a detailed breakdown of our model's predictions:
# 
# -True Negatives (top-left): 105 players were correctly predicted to have careers shorter than 5 years
# -False Positives (top-right): 22 players were incorrectly predicted to have careers of 5+ years when they actually had shorter careers
# -False Negatives (bottom-left): 94 players were incorrectly predicted to have careers shorter than 5 years when they actually had longer careers
# -True Positives (bottom-right): 114 players were correctly predicted to have careers of 5+ years
# 
# This matrix clearly illustrates our precision and recall metrics. The high precision (0.8382) is reflected by having relatively few false positives (22) compared to true positives (114). The lower recall (0.5481) is shown by the substantial number of false negatives (94), indicating that the model missed almost half of the players who actually had 5+ year careers. This confirms our model is conservative in predicting long careers - it's more likely to incorrectly classify a player as having a short career than to overestimate career longevity. For NBA talent development, this suggests the model could cause teams to miss out on potential long-term talent.

# ## Considerations
# 
# **What are some key takeaways that you learned from this lab?**
# 
# The key takeaways from this lab include:
# 
# 1.Naive Bayes provides a moderately effective solution for predicting NBA player career longevity, with 65.4% accuracy.
# 2.There's a significant imbalance between precision and recall in our model - it's highly precise (84%) but has lower recall (55%).
# 3.The model tends toward conservative predictions, missing many players who would actually have long careers (94 false negatives).
# 4.Evaluation metrics beyond accuracy are essential - the confusion matrix revealed important patterns that accuracy alone couldn't convey.
# 5.Feature engineering (done previously) is crucial, but our current feature set still leaves room for model improvement.
# 6.Model performance must be interpreted in the context of the business problem - in this case, a conservative model might cause teams to overlook potential long-term talent.
# 
# 
# **How would you present your results to your team?**
# 
# To present these results to my data science team, I would:
# 
# 1.Show the confusion matrix as a central visualization, highlighting the 94 false negatives as the major area for improvement.
# 2.Explain that our model achieves 65.4% accuracy with high precision (84%) but lower recall (55%).
# 3.Demonstrate that the model is better at confirming sure bets (players clearly destined for long careers) than identifying hidden gems.
# 4.Discuss potential reasons for the imbalance - perhaps certain important career longevity factors aren't captured in rookie year statistics.
# 5.Propose next steps, such as:
# 
# -Experimenting with different classification algorithms like Random Forest or XGBoost
# -Exploring additional features or transformations
# -Adjusting the classification threshold to improve recall at the cost of some precision
# -Implementing cost-sensitive learning to penalize false negatives more heavily
# 
# 
# **How would you summarize your findings to stakeholders?**
# 
# Executive Summary: NBA Player Career Longevity Prediction
# 
# Our Naive Bayes model provides NBA management with a valuable tool for talent investment decisions, accurately predicting 65% of rookie players' long-term potential. When the model identifies a player as a 5+ year prospect, you can trust this assessment with 84% confidence, allowing for more informed contract and development decisions.
# 
# However, the model currently overlooks nearly half of players who eventually develop long careers, as shown by the 94 false negatives in our analysis. This conservative approach means the system excels at confirming obvious talent but may miss borderline players with potential for long-term value.
# 
# We recommend integrating this model as one component of your talent evaluation process, particularly for validating high-confidence prospects. With your ongoing feedback and additional performance metrics, we can refine the model to better identify those hidden gems who could become valuable long-term assets for your organization.
# 
# 

# **Congratulations!** You've completed this lab. However, you may not notice a green check mark next to this item on Coursera's platform. Please continue your progress regardless of the check mark. Just click on the "save" icon at the top of this notebook to ensure your work has been logged
