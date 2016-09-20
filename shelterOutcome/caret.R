library(plyr)
library(e1071)
library(caret)

full.oneHotEncoded <- sparse.model.matrix(~.-1, data = full[, features])
x_train.oneHotEncoded <- as.matrix(full.oneHotEncoded[1:26729,])
x_test.oneHotEncoded <- as.matrix(full.oneHotEncoded[26730:nrow(full),])

fitControl <- trainControl(
  method = "cv", 
  number = 3, 
  repeats = 2,
  classProbs=TRUE,
  search = "random"
  )

model <- train(
  x=x_train.oneHotEncoded,
  y=as.factor(train$OutcomeType),
  method = "rf",
  trControl = fitControl
  )