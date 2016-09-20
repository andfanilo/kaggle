library(xgboost)
library(rBayesianOptimization)
library(data.table)

# Encode the data
full.oneHotEncoded <- sparse.model.matrix(~.-1, data = full[, features])
x_train.oneHotEncoded <- full.oneHotEncoded[1:26729,]
x_test.oneHotEncoded <- full.oneHotEncoded[26730:nrow(full),]

xgb_train <- xgb.DMatrix(data = x_train.oneHotEncoded, label=y_train)
xgb_test <- xgb.DMatrix(data = x_test.oneHotEncoded)

# Baseline cv test
#cv <- xgb.cv(data=xgb_train, 
#      nfold=5, 
#      nround=100, 
#      objective='multi:softprob', 
#      num_class=5,
#      max_depth=9,
#      min_child_weight=1,
#      colsample_bytree=0.85,
#      eval_metric='mlogloss'
#)

cv_folds <- KFold(y_train, 
                  nfolds = 3,
                  stratified = TRUE, 
                  seed = 0)

# Prepare function to optimize.
# We optimize Cross val over different xgboost hyperparams
xgb_cv_bayes <- function(max.depth, 
                         subsample,
                         colsample_bytree
                         ) {
  cv <- xgb.cv(params = list(max_depth = max.depth,
                             subsample = subsample, 
                             colsample_bytree = colsample_bytree,
                             objective = "multi:softprob",
                             eval_metric = "mlogloss",
                             num_class=5),
               data = xgb_train, 
               nround = 125,
               folds = cv_folds, 
               prediction = TRUE, 
               showsd = TRUE,
               #early.stop.round = 5,
               eta = 0.1,
               maximize = TRUE, 
               verbose = 1)
  list(Score = -cv$dt[, min(test.mlogloss.mean)], # Bayesian opti maximises --> negative value
       Pred = cv$pred)
}

# Begin exploration of the best hyperparameter values
OPT_Res <- BayesianOptimization(xgb_cv_bayes,
                                bounds = list(max.depth = c(6L, 10L),
                                              subsample = c(0.5, 0.9),
                                              colsample_bytree = c(0.5, 0.9)
                                              ),
                                init_points = 10, 
                                n_iter = 20,
                                acq = "ucb", 
                                kappa = 2.576, # try to alter for more explo
                                eps = 1.0,
                                verbose = TRUE
)

ensemble.history <- data.table(OPT_Res$History)

# let's keep only 50 % of the best
#summary(ensemble.history)
ensemble.history.filtered <- ensemble.history[ensemble.history$Value > median(ensemble.history$Value),]

# let's train all corresponding bst's to useful hyperparams
xgb_predict <- function(max.depth, 
                        subsample,
                        colsample_bytree) {
  bst <- xgboost(xgb_train, 
                  y_train, 
                  nrounds=125, 
                  objective='multi:softprob',
                  num_class=5, 
                  eval_metric='mlogloss',
                  eta = 0.1,
                  max_depth=max.depth,
                  subsample = subsample,
                  colsample_bytree=colsample_bytree
  )
  predict(bst, xgb_test)
}

# Make all predictions
all_predictions <- apply(ensemble.history.filtered, 1, 
      function(x) xgb_predict(x['max.depth'],x['subsample'],x['colsample_bytree'])
      )

all_predictions_means <- rowMeans(all_predictions)

# reshape predictions
xgb_preds <- data.frame(t(matrix(all_predictions_means, nrow=5, ncol=length(all_predictions_means)/5)))

# name columns
colnames(xgb_preds) <- c('Adoption', 'Died', 'Euthanasia', 'Return_to_owner', 'Transfer')

# attach ID column
xgb_preds['ID'] <- test['ID']

write.csv(xgb_preds[, c('ID', 'Adoption', 'Died', 'Euthanasia', 'Return_to_owner', 'Transfer')], 'data/submission.csv', row.names = FALSE)
