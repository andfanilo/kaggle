library(data.table)
library(FeatureHashing)
library(xgboost)
library(dplyr)
library(Matrix)

# Source : https://www.kaggle.com/raddar/predicting-red-hat-business-value/0-98-xgboost-on-sparse-matrix/

### Read data
train=fread('data/act_train.csv') #%>% as.data.frame()
test=fread('data/act_test.csv') #%>% as.data.frame()

### People data frame
people=fread('data/people.csv') #%>% as.data.frame()
people$char_1<-NULL #unnecessary duplicate to char_2
names(people)[2:length(names(people))]=paste0('people_',names(people)[2:length(names(people))])

## Transform logical cols to numbers
p_logi <- names(people)[which(sapply(people, is.logical))]
for (col in p_logi) set(people, j = col, value = as.numeric(people[[col]]))

## Reducing group_1 dimension
people$people_group_1[people$people_group_1 %in% names(which(table(people$people_group_1)==1))]='group unique'

## Reducing char_10 dimension
unique.char_10=
  rbind(
    select(train,people_id,char_10),
    select(train,people_id,char_10)) %>% group_by(char_10) %>% 
  summarize(n=n_distinct(people_id)) %>% 
  filter(n==1) %>% 
  select(char_10) %>%
  as.matrix() %>% 
  as.vector()

train$char_10[train$char_10 %in% unique.char_10]='type unique'
test$char_10[test$char_10 %in% unique.char_10]='type unique'

### Merge things
d1 <- merge(train, people, by = "people_id", all.x = T)
d2 <- merge(test, people, by = "people_id", all.x = T)
Y <- d1$outcome
d1$outcome <- NULL
row.train=nrow(train)

gc()

D=rbind(d1,d2)
D$i=1:dim(D)[1]

test_activity_id=test$activity_id
rm(train,test,d1,d2);gc()

char.cols=c('activity_category','people_group_1',
            'char_1','char_2','char_3','char_4',
            'char_5','char_6','char_7','char_8',
            'char_9','char_10', 'people_char_2','people_char_3',
            'people_char_4','people_char_5','people_char_6',
            'people_char_7','people_char_8','people_char_9')

for (f in char.cols) {
  if (class(D[[f]])=="character") {
    levels <- unique(c(D[[f]]))
    D[[f]] <- as.numeric(factor(D[[f]], levels=levels))
  }
}

D.sparse=
  cBind(sparseMatrix(D$i,D$activity_category),
        sparseMatrix(D$i,D$people_group_1),
        sparseMatrix(D$i,D$char_1),
        sparseMatrix(D$i,D$char_2),
        sparseMatrix(D$i,D$char_3),
        sparseMatrix(D$i,D$char_4),
        sparseMatrix(D$i,D$char_5),
        sparseMatrix(D$i,D$char_6),
        sparseMatrix(D$i,D$char_7),
        sparseMatrix(D$i,D$char_8),
        sparseMatrix(D$i,D$char_9),
        sparseMatrix(D$i,D$people_char_2),
        sparseMatrix(D$i,D$people_char_3),
        sparseMatrix(D$i,D$people_char_4),
        sparseMatrix(D$i,D$people_char_5),
        sparseMatrix(D$i,D$people_char_6),
        sparseMatrix(D$i,D$people_char_7),
        sparseMatrix(D$i,D$people_char_8),
        sparseMatrix(D$i,D$people_char_9)
  )

D.sparse=
  cBind(D.sparse,
        D$people_char_10,
        D$people_char_11,
        D$people_char_12,
        D$people_char_13,
        D$people_char_14,
        D$people_char_15,
        D$people_char_16,
        D$people_char_17,
        D$people_char_18,
        D$people_char_19,
        D$people_char_20,
        D$people_char_21,
        D$people_char_22,
        D$people_char_23,
        D$people_char_24,
        D$people_char_25,
        D$people_char_26,
        D$people_char_27,
        D$people_char_28,
        D$people_char_29,
        D$people_char_30,
        D$people_char_31,
        D$people_char_32,
        D$people_char_33,
        D$people_char_34,
        D$people_char_35,
        D$people_char_36,
        D$people_char_37,
        D$people_char_38,
        D$binay_sum)

train.sparse=D.sparse[1:row.train,]
test.sparse=D.sparse[(row.train+1):nrow(D.sparse),]

# Hash train to sparse dmatrix X_train
dtrain  <- xgb.DMatrix(train.sparse, label = Y)
dtest  <- xgb.DMatrix(test.sparse)

# XGBoost + Grid search-----------------------------------------

# Source: http://datascience.stackexchange.com/questions/9364/hypertuning-xgboost-parameters

set.seed(120)

# train & tune --skipped--
params_grid <- data.frame(shrinkage=c(0.04, 0.03, 0.03, 0.03, 0.02),
                      rounds = c(140, 160, 170, 140, 180),
                      depth = c(8, 7, 9, 10, 10),
                      gamma = c(0, 0, 0, 0, 0),
                      min.child = c(5, 5, 5, 5, 5),
                      colsample.bytree = c(0.7, 0.6, 0.65, 0.6, 0.85),
                      subsample = c(1, 0.9, 0.95, 1, 0.6))

# generate final prediction -- bag of 50 models --
models = 5
repeats = 10
preds  = rep(0, nrow(dtest))
for (j in 1:repeats) {
  for (i in 1:models) {
    set.seed(j * 1000 + i * 100)
    print(paste0(c("Position ", i, ";", j), collapse = ''))
    xgboost.mod = xgboost(
      data = dtrain,
      objective = "binary:logistic",
      eval_metric = "auc",
      booster = "gblinear",
      nrounds = params_grid$rounds[i],
      max.depth = params_grid$depth[i],
      eta = params_grid$shrinkage[i],
      subsample = params_grid$subsample[i],
      colsample_bytree = params_grid$colsample.bytree[i],
      gamma = params_grid$gamma[i],
      min.child.weight = params_grid$min.child[i],
      print.every.n = 50
    )
    preds = preds + predict(xgboost.mod, dtest)
  }
}

predictions_raddar =  preds/(models*repeats)
xgb_preds_raddar <- data.frame(activity_id = test_activity_id, outcome = predictions_raddar)
write.csv(xgb_preds_raddar, file = "submission/raddar_xgb_bagging_submission.csv", row.names = F)

# # set up the cross-validated hyper-parameter search
# #xgb_grid = expand.grid(
# #  ntrees = c(100, 300, 500),
# #  subsample = c(0.5, 0.75, 1.0), 
# #  colsample_bytree = c(0.4, 0.7, 1.0),
# #  max_depth = c(4, 7, 10)
# #)
# 
# xgb_grid = expand.grid(
#   eta = c(0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3)
# )
# 
# apply(xgb_grid, 1, function(parameterList){
#   
#   print(as.character(parameterList))
#   
#   #Extract Parameters to test
#   currentETA <- parameterList[["eta"]]
# 
#   bst <- xgb.train(data = dtrain,
#                   nrounds = 300,
#                   objective = "binary:logistic",
#                   eval_metric = "auc",
#                   booster = "gblinear",
#                   min_child_weight = 0,
#                   #eta = 2/currentNTrees,
#                   eta = currentETA,
#                   subsample = 0.7,
#                   colsample_bytree = 0.7,
#                   max_depth = 6,
#                   watchlist = list(train = dtrain),
#                   print.every.n = 50)
#   
#   predictions_raddar <- predict(bst, dtest)
#   xgb_preds_raddar <- data.frame(activity_id = test_activity_id, outcome = predictions_raddar)
#   paramsString <- paste(as.character(parameterList), collapse = '_')
#   write.csv(xgb_preds_raddar, paste(c("submission/submission_", paramsString, ".csv"), collapse = ""), row.names = F)
# })