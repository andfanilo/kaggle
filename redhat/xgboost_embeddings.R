library(data.table)
library(FeatureHashing)
library(xgboost)
library(dplyr)
library(Matrix)

python_embeddings <- fread("data/python_embeddings.csv")
python_act_ids <- fread("data/python_activity_ids.csv")

python_embeddings$activity_id <- python_act_ids$activity_id
rm(python_act_ids)

train=fread('data/act_train.csv')
test=fread('data/act_test.csv')

full  <- bind_rows(train, test)[, .(activity_id, outcome)]
full <- merge(full, python_embeddings, by = "activity_id")

train_indices <- !is.na(full$outcome)
test_indices <- is.na(full$outcome)
Y <- full$outcome[train_indices]
act_id_train <- full$activity_id[train_indices]
act_id_test <- full$activity_id[test_indices]
full[, outcome := NULL]
full[, activity_id := NULL]

full.oneHotEncoded <- sparse.model.matrix(~.-1, data = full)
train.oneHotEncoded <- full.oneHotEncoded[train_indices,]
test.oneHotEncoded <- full.oneHotEncoded[test_indices,]

gc()

xgb_train <- xgb.DMatrix(data = train.oneHotEncoded, label=Y)
xgb_test <- xgb.DMatrix(data = test.oneHotEncoded)

param <- list(objective = "binary:logistic", 
              eval_metric = "auc",
              booster = "gblinear", 
              eta = 0.02,
              subsample = 0.7,
              colsample_bytree = 0.7,
              min_child_weight = 0,
              max_depth = 10)

set.seed(120)
bst <- xgb.train(data = xgb_train, 
                param, nrounds = 50,
                watchlist = list(train = xgb_train),
                print_every_n = 10)

## Predict
preds <- predict(bst, xgb_train)
xgb_preds_raddar <- data.frame(activity_id = act_id_train, outcome = preds)
write.csv(xgb_preds_raddar, file = "submission/embeddings_train.csv", row.names = F)