library(dplyr)
library(xgboost)
library(rpart)
library(Matrix)
library(caret)
library(FeatureHashing)

dmerge <- function(x, y) left_join(x, y, by = "device_id")
toStr  <- function(x) paste(x, collapse = ",")

phone_brand_device_model <- phone_brand_device_model[ , .(brands = toStr(phone_brand), models = toStr(device_model)), by = device_id]

full <- bind_rows(label_train, label_test)
full_device <- dmerge(full, phone_brand_device_model)[, c("device_id", "brands", "models", "group")]
train_device <- full_device[!is.na(full_device$group), c("brands", "models")]
test_device <- full_device[is.na(full_device$group), c("brands", "models")]

# FeatureHash to sparse matrix
b <- hash.size(full_device)
f <- ~ split(brands, delim = ",") + split(models, delim = ",") - device_id
train_device_oneHotEncoded <- hashed.model.matrix(f, train_device, b, create.mapping = T)
test_device_oneHotEncoded <- hashed.model.matrix(f, test_device, b, create.mapping = T)

# Set a random seed
set.seed(1337)

xgb_train <- xgb.DMatrix(
  data = train_device_oneHotEncoded, 
  label = y_train
  )
xgb_test <- xgb.DMatrix(data = test_device_oneHotEncoded)

#xgb.cv(data=xgb_train, 
#       nfold=3, 
#       nround=100,
#       objective='multi:softprob',
#       num_class=num_class,
#       max_depth=9,
#       min_child_weight=1,
#       colsample_bytree=0.85,
#       eval_metric='mlogloss'
#       )

# quick xgboost
bst <- xgboost(xgb_train, 
               nrounds=100, 
               objective='multi:softprob',
               num_class=num_class, 
               eval_metric='mlogloss',
               early.stop.round=TRUE,
               max_depth=9
)

# make predictions
predictions <- predict(bst, xgb_test)

# reshape predictions
xgb_preds <- data.frame(t(matrix(predictions, nrow=num_class, ncol=length(predictions)/num_class)))
names(xgb_preds) <- submission_header[-1]
xgb_preds['device_id'] <- full_device[is.na(full_device$group),]$device_id

# name columns
#View(xgb_preds[submission_header])
write.csv(xgb_preds[, submission_header], 'data/submission.csv', row.names = FALSE)
