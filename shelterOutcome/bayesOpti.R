library(xgboost) # ...guess what
library(rBayesianOptimization)

# We are going to use Bayesian optimization to get best set of hyperparams for xgboost

# Set a random seed
set.seed(731)

full.oneHotEncoded <- sparse.model.matrix(~.-1, data = full[, features])
x_train.oneHotEncoded <- full.oneHotEncoded[1:26729,]
x_test.oneHotEncoded <- full.oneHotEncoded[26730:nrow(full),]

xgb_train <- xgb.DMatrix(data = x_train.oneHotEncoded, label=y_train)
xgb_test <- xgb.DMatrix(data = x_test.oneHotEncoded)

cv_folds <- KFold(y_train, 
                  nfolds = 10,
                  stratified = TRUE, 
                  seed = 0)

xgb_cv_bayes <- function(max.depth, 
                         min_child_weight, 
                         subsample,
                         gamma,
                         colsample_bytree,
                         eta) {
  cv <- xgb.cv(params = list(max_depth = max.depth,
                             min_child_weight = min_child_weight,
                             subsample = subsample, 
                             gamma = gamma,
                             colsample_bytree = colsample_bytree,
                             eta = eta,
                             objective = "multi:softprob",
                             eval_metric = "mlogloss",
                             num_class=5),
               data = xgb_train, 
               nround = 100,
               folds = cv_folds, 
               prediction = TRUE, 
               showsd = TRUE,
               early.stop.round = 5, 
               maximize = TRUE, 
               verbose = 0)
  list(Score = cv$dt[, max(test.mlogloss.mean)],
       Pred = cv$pred)
}

OPT_Res <- BayesianOptimization(xgb_cv_bayes,
                                bounds = list(max.depth = c(3L, 10L),
                                              min_child_weight = c(1L, 10L),
                                              subsample = c(0.6, 1.0),
                                              gamma = c(0.0, 0.5),
                                              colsample_bytree = c(0.6, 1.0),
                                              eta = c(0.01, 0.3)),
                                init_points = 10, 
                                n_iter = 30,
                                acq = "ucb", 
                                kappa = 2.576, 
                                eps = 0.0,
                                verbose = TRUE)