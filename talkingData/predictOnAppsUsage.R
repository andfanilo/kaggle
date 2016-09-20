library(xgboost)
library(dplyr)
library(rpart)
library(Matrix)
library(caret)
library(FeatureHashing)
library(lubridate)
library(hashmap)

daily_mean_label <- fread("data/daily_mean_label_device.csv",
                          colClasses=c("character","double","double"))

daily_mean_label <- dcast(daily_mean_label, device_id ~ label_id, value.var = "mean", fill = 0.0)

full <- bind_rows(label_train, label_test)[,.(device_id, group)]
full_apps <- merge(full, daily_mean_label, by="device_id", all.x = T)

train_apps <- full_apps[!is.na(full_apps$group),]
y_train <- as.numeric(as.factor(train_apps$group)) - 1
train_apps[,group:=NULL]#[, device_id:=NULL]

test_apps <- full_apps[is.na(full_apps$group),][,group:=NULL]
d_id <- test_apps$device_id
test_apps#[, device_id:=NULL]

# Set a random seed
set.seed(1337)
 
# Dimension reduction using PCA https://www.r-bloggers.com/computing-and-visualizing-pca-in-r/
#trans <- preProcess(train_apps[complete.cases(train_apps),]  + (rnorm(nrow(train_apps)) / 1000) , method="pca")
#pca <- predict(trans, train_apps)
#pca <- prcomp(train_apps)
#train_apps_pca <- predict(pca, train_apps)
#test_apps_pca <- predict(pca, test_apps)
#plot(pca, type = "l")
#summary(pca)

#xgb_train <- xgb.DMatrix(
#  data = as.matrix(train_apps), 
#  label = y_train,
#  missing = NA
#)
#xgb_test <- xgb.DMatrix(data = as.matrix(test_apps), missing = NA)

## quick xgboost
#bst <- xgboost(xgb_train, 
#               nrounds=25, 
#               objective='multi:softprob',
#               num_class=num_class, 
#               eval_metric='mlogloss',
#               early.stop.round=TRUE,
#               max_depth=6
#)

#xgb.importance(feature_names = names(train_apps), model = bst)

#
## make predictions
#predictions <- predict(bst, xgb_test)
#
## reshape predictions
#xgb_preds <- data.frame(t(matrix(predictions, nrow=num_class, ncol=length(predictions)/num_class)))
#names(xgb_preds) <- submission_header[-1]
#xgb_preds['device_id'] <- d_id
#
## name columns
##View(xgb_preds[submission_header])
#write.csv(xgb_preds[, submission_header], 'data/submission.csv', row.names = FALSE)