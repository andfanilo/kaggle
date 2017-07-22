library(xgboost)

# Kinda sources from https://www.kaggle.com/cartographic/bosch-production-line-performance/bish-bash-xgboost

# Train model ---- 

dmodel <- xgb.DMatrix(train_numeric[model,2:969], label = y[model])
dvalid <- xgb.DMatrix(train_numeric[valid,2:969], label = y[valid])
dtest <- xgb.DMatrix(test_numeric[,2:969])

param <- list(objective = "binary:logistic",
              eval_metric = "auc",
              eta = 0.01,
              base_score = 0.005, # helps with imbalanced class
              col_sample = 0.5)

m1 <- xgb.train(data = dmodel, param, nrounds = 20,
                watchlist = list(mod = dmodel, val = dvalid))

#imp <- xgb.importance(model = m1, feature_names = train_numeric@Dimnames[[2]])
#head(imp, 100)

# Determine threshold by Matthews Coefficient ---- 
mc <- function(actual, predicted) {
  
  tp <- as.numeric(sum(actual == 1 & predicted == 1))
  tn <- as.numeric(sum(actual == 0 & predicted == 0))
  fp <- as.numeric(sum(actual == 0 & predicted == 1))
  fn <- as.numeric(sum(actual == 1 & predicted == 0))
  
  numer <- (tp * tn) - (fp * fn)
  denom <- ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ^ 0.5
  
  numer / denom
}

valid_preds_num <- predict(m1, dvalid)
matt <- data.table(thresh = seq(0.990, 0.999, by = 0.001))
matt$scores <- sapply(matt$thresh, FUN =
                        function(x) mc(y[valid], (valid_preds_num > quantile(valid_preds_num, x)) * 1))
best_thresh <- matt$thresh[which(matt$scores == max(matt$scores))]

# Output result ----
test_preds_num <- predict(m1, dtest)
options(scipen = 999) # remove exponential form from file -_-
sub   <- data.table(Id = as.numeric(test_ids),
                    Response = (test_preds_num > quantile(test_preds_num, best_thresh)) * 1,
                    Probs = test_preds_num)

write.csv(sub[, .(Id, Response)], "submission/quickNumericXGBoost.csv", row.names = F)
write.csv(sub[, .(Id, Probs)], "submission/quickNumericXGBoost_probs.csv", row.names = F)
