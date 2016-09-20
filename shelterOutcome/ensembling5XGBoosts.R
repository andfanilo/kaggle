library(xgboost)
library(caret)

# Set a random seed
set.seed(731)

full.oneHotEncoded <- sparse.model.matrix(~.-1, data = full[, features])
x_train.oneHotEncoded <- full.oneHotEncoded[1:26729,]
x_test.oneHotEncoded <- full.oneHotEncoded[26730:nrow(full),]

xgb_train <- xgb.DMatrix(data = x_train.oneHotEncoded, label=y_train)
xgb_test <- xgb.DMatrix(data = x_test.oneHotEncoded)

bst1 <- xgboost(xgb_train, 
               y_train, 
               nrounds=125, 
               objective='multi:softprob',
               num_class=5, 
               eval_metric='mlogloss',
               early.stop.round=TRUE,
               max_depth=7,
               subsample = 0.75,
               colsample_bytree=0.85,
               eta = 0.1
)

bst2 <- xgboost(xgb_train, 
                y_train, 
                nrounds=125, 
                objective='multi:softprob',
                num_class=5, 
                eval_metric='mlogloss',
                early.stop.round=TRUE,
                max_depth=6,
                subsample = 0.85,
                eta = 0.1,
                colsample_bytree=0.75
)

bst3 <- xgboost(xgb_train, 
                y_train, 
                nrounds=125, 
                objective='multi:softprob',
                num_class=5, 
                eval_metric='mlogloss',
                early.stop.round=TRUE,
                max_depth=8,
                subsample = 0.85,
                eta = 0.1,
                colsample_bytree=0.75
)

bst4 <- xgboost(xgb_train, 
                y_train, 
                nrounds=125, 
                objective='multi:softprob',
                num_class=5, 
                eval_metric='mlogloss',
                early.stop.round=TRUE,
                max_depth=9,
                subsample = 0.55,
                eta = 0.1,
                colsample_bytree=0.65
)

bst5 <- xgboost(xgb_train, 
                y_train, 
                nrounds=125, 
                objective='multi:softprob',
                num_class=5, 
                eval_metric='mlogloss',
                early.stop.round=TRUE,
                max_depth=10,
                subsample = 0.55,
                eta = 0.1,
                colsample_bytree=0.55
)


# make predictions
predictions1 <- predict(bst1, xgb_test)
predictions2 <- predict(bst2, xgb_test)
predictions3 <- predict(bst3, xgb_test)
predictions4 <- predict(bst4, xgb_test)
predictions5 <- predict(bst5, xgb_test)

predictions <- 0.2 * (predictions1+predictions2+predictions3+predictions4+predictions5)

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

write.csv(xgb_preds[, c('ID', 'Adoption', 'Died', 'Euthanasia', 'Return_to_owner', 'Transfer')], 'data/submission.csv', row.names = FALSE)
