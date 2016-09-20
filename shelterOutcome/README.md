# Kaggle Shelter outcome competition 

_URL : https://www.kaggle.com/c/shelter-animal-outcomes_

This repo contains a set of R scripts I wrote for the Shelter Animal Outcomes.

### Disclaimer 

This is my first Kaggle competition entry. I originally used the Anaconda stack, but then decided to try R for the first time 
and I was quite happy with the result (made it to top 15% without copying/pasting too much code from the forums). 

I have put around 20+ hours into the competition, a good third of it on learning the basics of R and exploring its packages. 
I did not have time to implement a sound cross-validation strategy, nor beautiful feature engineering graphs,  
focusing more of my efforts on producing a result and submitting to Kaggle.

### preprocess.R

This should be the first script you run. It preprocesses the features (extracting breed, color, inferring ages and stuff) 
then builds the dataframes tp use for training. If I recall it heavily relies on https://www.kaggle.com/mrisdal/shelter-animal-outcomes/quick-dirty-randomforest.

### xgboost.R

First time using XGBoost :p convert to sparse matrix https://github.com/dmlc/xgboost/blob/master/R-package/demo/create_sparse_matrix.R for
one hot encoding variables, then xgb.cv, train and predict.

### randomForest.R

Trying the randomForest package.

### caret.R

I wanted to try ensembling caret randomForest and XGBoost models (https://github.com/dmlc/xgboost/blob/master/R-package/demo/caret_wrapper.R),
but I think randomForest took too long so I let it slide.

### dimReduction.R

With all of this categorical data, I remembered one of my colleagues talking about FactorMineR so wanted to try feature engineering using it.
Too bad I didn't go that far, should put more efforts into it next time.

### quickEnsembling.R

I read about ensembling (http://mlwave.com/kaggle-ensembling-guide/) and I wanted to try _(who said I should focus on feature engineering before ensembling for better results ;) ?)_

This is my first try, averaging the predictions of 1 randomForest and 1 XGBoost. 
I got 20 places better I think, so I decided to focus on ensembling/stacking for the rest of the competition, instead of crossval strategy or feature engineering.

### logitEnsembling.R

In my head, stacking a logit regression over XGBoost and randomForest models sounded cool ! In practice, the results were a bit terrible, 
I amsure there are many reasons for it but I willl push that analysis for another day.

### ensembling5XGBoost.R

Just averaging 5 XGBoost with different meta params. I think I got those meta params from a script but am too lazy to go found it. 

Gave me a good bump on the leaderboard, to about 18%, that remotivated me since I had been stalling at 26% for 2 weeks. 
So my next approach was on looking for the best hyperparameters for xgboost models to average.

### bayesOpti.R

So I was casually reading on the Automatic Machine Learning Challenge, when I came accross Bayesian optimization of hyperparameters.

I decided to give it a try, I wanted to configure it to explore the hyperparameter space, to retrieve a set of good hyperparams instead of looking for the very best one.

### bayesOptiEnsembling.R

When the exploration finishes, I have a set of good hyperparameters, so I trained their corresponding XGBoost models and then averaged all of their predictions. 
That got me to 14% on the leaderboard and I was satisfied with it. 
