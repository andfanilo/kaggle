# Source : http://www.kdnuggets.com/2015/06/ensembles-kaggle-data-science-competition-p2.html

#Let’s say you want to do 2-fold stacking:
#  
#  Split the train set in 2 parts: train_a and train_b
#Fit a first-stage model on train_a and create predictions for train_b
#Fit the same model on train_b and create predictions for train_a
#Finally fit the model on the entire train set and create predictions for the test set.
#Now train a second-stage stacker model on the probabilities from the first-stage model(s).
#A stacker model gets more information on the problem space by using the first-stage predictions as features, 
# than if it was trained in isolation.

library(data.table)
library(FeatureHashing)
library(xgboost)
library(dplyr)
library(Matrix)
library(caret)

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

### Go stacking generalization -----

param <- list(objective = "binary:logistic", 
              eval_metric = "auc",
              booster = "gblinear", 
              eta = 0.02,
              subsample = 0.7,
              colsample_bytree = 0.7,
              min_child_weight = 0,
              max_depth = 10)

set.seed(120)
nrounds <- 305

### Split the train set in 3 parts: train_a and train_b
folds <- createFolds(y = Y, k = 3, list = T)
Y1 <- Y[folds[[1]]]
Y2 <- Y[folds[[2]]]
Y3 <- Y[folds[[3]]]
n_Y1 <- Y[-folds[[1]]]
n_Y2 <- Y[-folds[[2]]]
n_Y3 <- Y[-folds[[3]]]
dtrain1  <- xgb.DMatrix(train.sparse[folds[[1]],], label = Y1)
dtrain2  <- xgb.DMatrix(train.sparse[folds[[2]],], label = Y2)
dtrain3  <- xgb.DMatrix(train.sparse[folds[[3]],], label = Y3)
n_dtrain1  <- xgb.DMatrix(train.sparse[-folds[[1]],], label = n_Y1)
n_dtrain2  <- xgb.DMatrix(train.sparse[-folds[[2]],], label = n_Y2)
n_dtrain3  <- xgb.DMatrix(train.sparse[-folds[[3]],], label = n_Y3)

model1 <- xgb.train(
  data = n_dtrain1, 
  param, 
  nrounds = nrounds,
  watchlist = list(train = n_dtrain1, test= dtrain1),
  print_every_n = 10
)
predictions1 <- setNames(as.data.table(cbind((1:length(Y))[folds[[1]]], predict(model1, dtrain1))), c("id", "pred"))

model2 <- xgb.train(
  data = n_dtrain2, 
  param, 
  nrounds = nrounds,
  watchlist = list(train = n_dtrain2, test= dtrain2),
  print_every_n = 10
)
predictions2 <- setNames(as.data.table(cbind((1:length(Y))[folds[[2]]], predict(model2, dtrain2))), c("id", "pred"))

model3 <- xgb.train(
  data = n_dtrain3, 
  param, 
  nrounds = nrounds,
  watchlist = list(train = n_dtrain3, test= dtrain3),
  print_every_n = 10
)
predictions3 <- setNames(as.data.table(cbind((1:length(Y))[folds[[3]]], predict(model3, dtrain3))), c("id", "pred"))

### Finally fit the model on the entire train set and create predictions for the test set.
model_l1 <- xgb.train(
  data = xgb.DMatrix(train.sparse, label = Y), 
  param, 
  nrounds = nrounds,
  print_every_n = 10
)
predictions_l1_test <- setNames(as.data.table(predict(model_l1, xgb.DMatrix(test.sparse))), c("pred"))

### Create training set of first layer
t <- do.call("rbind", list(predictions1, predictions2, predictions3))[order(id),]
t$outcome <- Y
t$id <- NULL

### Now train a second-stage stacker model on the probabilities from the first-stage model(s).
model <- glm(outcome ~ ., data = t, family = "binomial")

#data.frame(outcome=Y, prediction=predict(model, t, type = "response"))
predictions_raddar <- predict(model, predictions_l1_test, type = "response")
xgb_preds_raddar <- data.frame(activity_id = test_activity_id, outcome = predictions_raddar)
write.csv(xgb_preds_raddar, file = "data/raddar_submission.csv", row.names = F)
