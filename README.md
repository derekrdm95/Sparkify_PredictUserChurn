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
Here we use **F1 score** to evaluate and select the model. First, this is a relatively small dataset, we have very few users that actually churn from our service. Besides, the reason is that overall metrics such as accuracy is not very helpful in the case of user churn. Churn rate usually not very high acroos the whole population. Here we would like to focus more on the precision (those who were predicted churn would truly churn in our case) and recall in our scenario. Therefore, f1 score is a better metrics reflecting what we care about.

### Feature Engineering and Preprocessing Step
After the preliminary exploratory analysis, **I started to create some of the features, aggregating all the log activities to userID level using pyspark**. Some features include:

- listen_time: how long do they listen to the app
- num_song: how many sons they listend to
- numb_thumb_down, num_thumb_up: how many times they thumb-up/down (interaction with the songs)
- add_to_playlist: how many times they add the songs to their playlist (the quality of our recommendation system)
- lt: their lifetime (most recent activity - acquisition time)
- add_friend (how many friends they added)
- help (if they clicked for help)
- rolladvert (if they roll advertisement)

For **preprosessing** step: 
- I first deleted all the log activities with missing user_id.
- For all the numeric feature, I combine them altogether by user_id and than normalize all of them

### Implementation
**Step1:** Split the dataset into 80% training set, 20% valdiation set
**Step2:** Train the training set on three different models: logistic regression, svm and gradient boosting tree
***Initial Parameters:***
- logit = LogisticRegression(maxIter=10,regParam=0.0)
- gbt = GBTClassifier(maxDepth=5,maxIter=10,seed=42)
- svm = LinearSVC(maxIter=10, regParam=0.01)
**Step3:** Evaluate the three models use the 3-fold cross validation
***Respective f1 score:***
- logistic regression (f1 score 0.7146)
- gradient boosted tree (best f1 score 0.7160 )
- supported vector machine (best f1 score 0.6780)

### Refinement
Here I tried multiple grid-searches, tuning different hyperparamater:
- logistic regression (logit.regParam, [0.0, 0.05, 0.1])
- gradient boosted tree (gbt.maxDepth, [5, 10])
- supported vector machine (svm.regParam, [0.01, 0.05, 0.5])
But several hyperparameters have been tried, it seems that still the original one is the best.

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

**Benchmark of our base model**
The most basic estimates is to predict all the user to not churn in this case, our accuracy would be 99/448 = 0.221
Therefore, our final model greatly improved the prediction.

### Improvement
- Feature Engineering: There are more interesting features that could be included in the models such as **whether they are free/paid users; whether they swtiched their services; what is their frequency and interval of using our product.** 
- Tunning Hyperparameter: Training GBT on a fairly large dataset could take quite a long time. Given more time, more and wider range of hyperparameters could be tested on for the best model such as **lossType, maxBins, minInfoGain**.

### Reflection
**Summary of Solution**
To summarize the whole analysis, I built a machine learning model using pyspark to predict the user churn based on their log activity. By aggregating the data down to user-level engineering different features, training different machine learning models, tunining hyperparameter and cross validation to pick the best models, I finally picked the GBT models (maxDepth=5, maxIter=10) (Accuracy: 0.7816, F1 Score: 0.7416). Feature Importances were reported in the end.

I find the project especially interesting because it's my first time to use spark. And I realize how many data records of an indivdual could be. A user could have hundreds of log records and therefore distributed computing is necessary. However, after we aggregate the data to user level. The size of the user sample actually is not very big. Therefore, it rely on us to use more robust evaluation score (f1 score) to train the model.

