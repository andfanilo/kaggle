library(nnet)
library(glmnet)

labels <- c('Adoption', 'Died', 'Euthanasia', 'Return_to_owner', 'Transfer')

test_xgb <- xgb_preds[, labels]
test_rf <- rf_preds[, labels]

temp <- predict(bst, xgb_train)
train_xgb <- data.frame(t(matrix(temp, nrow=5, ncol=length(temp)/5)))
train_rf <- data.frame(predict(rf, full[1:26729, ], type='vote'))

colnames(train_rf) <- sapply(labels, function(x) paste("rf",x, sep=""))
colnames(train_xgb) <- sapply(labels, function(x) paste("xgb",x, sep=""))

colnames(test_rf) <- sapply(labels, function(x) paste("rf",x, sep=""))
colnames(test_xgb) <- sapply(labels, function(x) paste("xgb",x, sep=""))

train_merge_preds <- cbind(train_rf, train_xgb, train$OutcomeType)
test_merge_preds <- cbind(test_rf, test_xgb)

# nnet
#ensemble <- multinom(`train$OutcomeType` ~ (.)^2, data <- train_merge_preds)

#glmnet
ensemble <- glmnet(
  x=as.matrix(cbind(train_rf, train_xgb)), 
  y=y_train,
  family="multinomial"
  )

merge_preds <- predict(ensemble, test_merge_preds, "probs")
final_preds <- data.frame(t(matrix(merge_preds, nrow=5, ncol=length(merge_preds)/5)))
colnames(final_preds) <- c('Adoption', 'Died', 'Euthanasia', 'Return_to_owner', 'Transfer')
final_preds['ID'] <- test['ID']

write.csv(xgb_preds[, c('ID', 
                        'Adoption', 
                        'Died', 
                        'Euthanasia', 
                        'Return_to_owner', 
                        'Transfer')], 
          'data/submission.csv', row.names = FALSE)