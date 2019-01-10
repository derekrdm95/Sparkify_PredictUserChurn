# Sparkify_PredictUserChurn
Predicting User Churn with Sparkify JSON Log File

### Key Project Objective and Background
The dataset I worked is called Sparky, which is a streaming service user log file similar to Spotify. The aim of the project is to **build up a machine learning model to predict user churn (cancellation)** 

The whole analysis is specifically asked to perform using **PySpark**. Besides, the model was first built on a mini dataset. And then it can be deployed onto cloud platforms such as IBM Watson Studio or AWS. **Here all my analysis is based on the final computing output by using IBM Watson Studio.**

### Methodology
Please refer to my notebook file for detailed analysis, the whole analysis is divided into several parts
- Define Target Variable (churn) and Exploratory Analysis
- Feature Engineering (Create New Feature and Transformation)
- Modeling (Here we used three different models **logistic regression, SVM and gradient boosting tree**)
- Evaluation (Pick up the best model and report result on **validation set**)

### Conclusion
Among the three models we've been trained:
- logistic regression (best f1 score 0.7147)
- gradient boosted tree (best f1 score 0.7621 )
- supported vector machine (best f1 score 0.6781)

Therefore, we decided to choose **gradient boosted tree as the best model with tunned parameter (maxDepth=5, maxIter=10)**. And we fit our best model on the validation set. And we look at the feature importance as well.

Score on the final validation set
- Accuracy: 0.7816
- F1 Score: 0.7416
