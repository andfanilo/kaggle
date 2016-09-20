library(xgboost)
library(caret)

# Set a random seed
set.seed(731)

# so we need to convert factors now : https://github.com/dmlc/xgboost/blob/master/R-package/vignettes/discoverYourData.Rmd
# go one hot encoding

# Method 1 - better memory and speed
#full.oneHotEncoded <- sparse.model.matrix(~.-1, data = full[, features])
#x_train.oneHotEncoded <- full.oneHotEncoded[1:26729,]
#x_test.oneHotEncoded <- full.oneHotEncoded[26730:nrow(full),]

# Method 2 - worse speed & memory, but I get a veryyyyy small improv' xD
dummify <- dummyVars(~., x_train)
x_train.oneHotEncoded <- predict(dummify, x_train)
x_test.oneHotEncoded <- predict(dummify, x_test)

# Testing XGBoost

xgb_train <- xgb.DMatrix(data = x_train.oneHotEncoded, label=y_train)
xgb_test <- xgb.DMatrix(data = x_test.oneHotEncoded)

#xgb.cv(data=xgb_train, 
#      nfold=5, 
#      nround=100, 
#      objective='multi:softprob', 
#      num_class=5,
#      max_depth=9,
#      min_child_weight=1,
#      colsample_bytree=0.85,
#      eval_metric='mlogloss')

# quick xgboost
#bst <- xgboost(xgb_train, 
#               y_train, 
#               nrounds=100, 
#               objective='multi:softprob',
#               num_class=5, 
#               eval_metric='mlogloss',
#               early.stop.round=TRUE,
#               max_depth=9,
#               min_child_weight=1,
#               colsample_bytree=0.85
#)

# bayesian optimised hyperparams xgboost
bst <- xgboost(xgb_train, 
               y_train, 
               nrounds=100, 
               objective='multi:softprob',
               num_class=5, 
               eval_metric='mlogloss',
               early.stop.round=TRUE,
               max_depth=5,
               subsample = 0.6895,
               gamma = 0.4920,
               eta = 0.29,
               min_child_weight=2,
               colsample_bytree=0.8352
)


# make predictions
predictions <- predict(bst, xgb_test)

# reshape predictions
xgb_preds <- data.frame(t(matrix(predictions, nrow=5, ncol=length(predictions)/5)))

# name columns
colnames(xgb_preds) <- c('Adoption', 'Died', 'Euthanasia', 'Return_to_owner', 'Transfer')

# attach ID column
xgb_preds['ID'] <- test['ID']

# quick peek - looks good
head(xgb_preds)

# features importance (first arg is vector of all feature names)
#importanceRaw <- xgb.importance(colnames(x_train.oneHotEncoded), model = bst)
#importanceClean <- importanceRaw[,`:=`(Cover=NULL, Frequence=NULL)]

#write.csv(xgb_preds[, c('ID', 'Adoption', 'Died', 'Euthanasia', 'Return_to_owner', 'Transfer')], 'data/submission.csv', row.names = FALSE)
