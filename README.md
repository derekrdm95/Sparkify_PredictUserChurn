# Sparkify_PredictUserChurn
Predicting User Churn with Sparkify JSON Log File

### Key Project Objective and Background
The dataset I worked is called Sparky, which is a streaming service user log file similar to Spotify. The aim of the project is to **build up a machine learning model to predict user churn (cancellation)** 

#### Strategy
In order to solve the problems. The whole analysis is performed using **PySpark**. We woud like to predict the **probability of user churn** There, our strategy to solve the problem is to firstly build and select supervised learning models on a mini dataset. And then it can be deployed onto cloud platforms such as IBM Watson Studio or AWS for a larger dataset. **Here all my analysis is based on the final computing output by using IBM Watson Studio.**

#### Library
The libraries I used in this analysis include 
- pyspark.sql; pyspark.ml
- re, datetime, numpy, pandas and matplotlib

#### File in the Repository
- README
- Jupytor Notebook of the Analysis

#### acknowledgement
The project and analysis was kindly provided by the Udacity Data Scientist Nano-degree project

### Methodology
Please refer to my notebook file for detailed analysis, the whole analysis is divided into several parts
- Define Target Variable (churn) and Exploratory Analysis
- Feature Engineering (Create New Feature and Transformation)
- Modeling (Here we used three different models **logistic regression, SVM and gradient boosting tree**)
- Evaluation (Pick up the best model and report result on **validation set**)

### Metrics For Evaluation and Justification
Here we use F1 score to evaluate and select the model. The reason is that overall accuracy such as accuracy is not very helpful in the case of user churn. Churn rate usually not very high acroos the whole population. Here we would like to focus more on the precision (those who were predicted churn would truly churn in our case) and recall in our scenario. Therefore, f1 score is a better metrics reflecting what we care about.


### Conclusion
Among the three models we've been trained:
- logistic regression (best f1 score 0.7147)
- gradient boosted tree (best f1 score 0.7621 )
- supported vector machine (best f1 score 0.6781)

For the **reason why GBT outperformed other models**, I think it is benefited from the boosting method. For each iteration, the algorithm would put more weight and train more rigoriously on those misclassified records. And the final model is the weighted average of multiple relatively week models. This method works very well in this case, especially we would like to correctly predict those user churn. 

Therefore, we decided to choose **gradient boosted tree as the best model with tunned parameter (maxDepth=5, maxIter=10)**. And we fit our best model on the validation set. And we look at the feature importance as well.

Score on the final validation set
- Accuracy: 0.7816
- F1 Score: 0.7416
