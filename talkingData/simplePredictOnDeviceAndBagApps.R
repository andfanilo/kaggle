library(dplyr)
library(xgboost)
library(rpart)
library(Matrix)
library(caret)
library(FeatureHashing)

dmerge <- function(x, y) left_join(x, y, by = "device_id")
toStr  <- function(x) paste(x, collapse = "|")

phone_brand_device_model <- phone_brand_device_model[!duplicated(phone_brand_device_model$device_id), ]

app_events <- app_events[ , .(apps = toStr(app_id)), by = event_id]

events <- merge(events, app_events, by = "event_id", all.x = T)
events <- events[ , .(apps = toStr(apps)), by = device_id]
##write.csv(events, 'data/better_events.csv', row.names = F)
rm(app_events)

#events <- fread('data/better_events.csv', colClasses=c("character","character"))

full <- bind_rows(label_train, label_test)
full_device <- dmerge(full, phone_brand_device_model)[, c("device_id", "phone_brand", "device_model", "group")]
events$device_id <- as.character(events$device_id)
full_device_apps <- dmerge(full_device, events)
train_device <- full_device_apps[!is.na(full_device_apps$group), c("device_id", "phone_brand", "device_model", "apps", "group")]
test_device <- full_device_apps[is.na(full_device_apps$group), c("device_id", "phone_brand", "device_model", "apps", "group")]

# FeatureHash to sparse matrix
#b <- hash.size(full_device_apps)
#f <- ~ phone_brand + device_model + split(apps, delim = "|") - device_id
#train_oneHotEncoded <- hashed.model.matrix(f, train_device, b, create.mapping = T)
#test_oneHotEncoded <- hashed.model.matrix(f, test_device, b, create.mapping = T)

# Set a random seed
set.seed(1337)

#xgb_train <- xgb.DMatrix(
#  data = train_oneHotEncoded, 
#  label = y_train
#)
#xgb_test <- xgb.DMatrix(data = test_oneHotEncoded)

#xgb.cv(data=xgb_train, 
#       nfold=3, 
#       nround=25,
#       objective='multi:softprob',
#       num_class=num_class,
#       max_depth=9,
#       min_child_weight=1,
#       colsample_bytree=0.85,
#       eval_metric='mlogloss'
#       )

# quick xgboost
#bst <- xgboost(xgb_train, 
#               nrounds=25, 
#               objective='multi:softprob',
#               num_class=num_class, 
#               eval_metric='mlogloss',
#               early.stop.round=TRUE,
#               max_depth=9
#)
#
#xgb.importance(feature_names = names(train_oneHotEncoded), model = bst)
#
## make predictions
#predictions <- predict(bst, xgb_test)
#
## reshape predictions
#xgb_preds <- data.frame(t(matrix(predictions, nrow=num_class, ncol=length(predictions)/num_class)))
#names(xgb_preds) <- submission_header[-1]
#xgb_preds['device_id'] <- full_device[is.na(full_device$group),]$device_id
#
## name columns
##View(xgb_preds[submission_header])
#write.csv(xgb_preds[, submission_header], 'data/submission.csv', row.names = FALSE)
