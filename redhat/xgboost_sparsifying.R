library(xgboost)
library(Matrix)

### Set a random seed
set.seed(731)

## Reduce group_1 (from Raddar)
full$group_1[full$group_1 %in% names(which(table(full$group_1)<10))]='group unique'

### One hot encoding
full.oneHotEncoded <- sparse.model.matrix(~.-1, data = full)

xgb_train <- xgb.DMatrix(data = full.oneHotEncoded[1:nrow(act_train),], label=y_train)
xgb_test <- xgb.DMatrix(data = full.oneHotEncoded[(nrow(act_train)+1):nrow(act_full),])

#xgb.cv(data=xgb_train, 
#       nfold = 5,
#       nrounds=50,
#       eval_metric = "auc",
#       objective='binary:logistic',
#       booster="gblinear",
#       max_depth=10,
#       subsample = 0.7,
#       colsample_bytree = 0.7,
#       min_child_weight = 0
#)

### Quick xgboost
bst <- xgboost(xgb_train, 
               y_train, 
               nrounds=300,
               eval_metric = "auc",
               objective='binary:logistic',
               booster='gblinear',
               eta = 0.02,
               max_depth=10,
               subsample = 0.7,
               colsample_bytree = 0.7,
               min_child_weight = 0
)

### Make predictions
predictions <- predict(bst, xgb_test)
xgb_preds <- data.frame(act_id_test, predictions)
names(xgb_preds) <- c('activity_id','outcome')

### if gbtree then this is usable to study features
#importance <- xgb.importance(feature_names = full.oneHotEncoded@Dimnames[[2]], model = bst)
#importanceRaw <- xgb.importance(feature_names = full.oneHotEncoded@Dimnames[[2]], model = bst, data = full.oneHotEncoded, label = y_train)
#importanceClean <- importanceRaw[,`:=`(Cover=NULL, Frequency=NULL)]

### End this
write.csv(xgb_preds, 'data/me_submission.csv', row.names = FALSE)