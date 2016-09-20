library(data.table)

train_raddar <- fread("submission/raddar_train.csv")
test_raddar <- fread("submission/raddar_test.csv")

train_embeds <- fread("submission/embeddings_train.csv")
test_embeds <- fread("submission/embeddings_test.csv")

train <- merge(train_raddar, train_embeds, by="activity_id")
test <- merge(test_raddar, test_embeds, by="activity_id")

acts=fread('data/act_train.csv')[, .(activity_id, outcome)]

train <- merge(train, acts, by="activity_id")

rm(acts)

act_id <- test$activity_id

train[, activity_id:=NULL]
test[, activity_id:=NULL]

model <- glm(outcome ~ ., data = train, family = "binomial")

predictions_raddar <- predict(model, test, type = "response")
xgb_preds_raddar <- data.frame(activity_id = act_id, outcome = predictions_raddar)
write.csv(xgb_preds_raddar, file = "data/stacking_submission.csv", row.names = F)
