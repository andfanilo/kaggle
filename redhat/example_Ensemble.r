## R version 3.1.0 (2014-04-10) -- "Spring Dance"
## Copyright (C) 2014 The R Foundation for Statistical Computing
## Platform: x86_64-w64-mingw32/x64 (64-bit)

rm(list=ls(all=TRUE))

##Loading Libraries

library(randomForest)
library(gbm)
library(xgboost)

##set the seed

set.seed(123)

## Read the train & test file 

train=read.csv("train.csv")
test=read.csv("test.csv")


# extract id
id.test = test$Id
test$Id = NULL
train$Id = NULL
n = nrow(train)

# extarct target
y = train$Hazard
train$Hazard = NULL

# replace factors with level mean hazard
for (i in 1:ncol(train)) {
  if (class(train[,i])=="factor") {
    mm <- aggregate(y~train[,i], data=train, mean)
    levels(train[,i]) <- as.numeric(mm[,2]) 
    levels(test[,i]) <- mm[,2] 
    train[,i] <- as.numeric(as.character(train[,i]))  
    test[,i] <- as.numeric(as.character(test[,i]))
  }
}


rf1=randomForest(log(y)~.,data=train,ntree=25,seed=343,metric="rmse")
rf2=randomForest( log(y)~.,data=train,ntree=25,seed=674,metric="rmse")
rf3=randomForest(log(y)~.,data=train,ntree=25,seed=856,metric="rmse")
rf4=randomForest(log(y)~.,data=train,ntree=25,seed=934,metric="rmse")
rf5=randomForest(log(y)~.,data=train,ntree=25,seed=564,metric="rmse")


pred1=predict(rf1,newdata=test,type="response")
pred2=predict(rf2,newdata=test,type="response")
pred3=predict(rf3,newdata=test,type="response")
pred4=predict(rf4,newdata=test,type="response")
pred5=predict(rf5,newdata=test,type="response")

pred_random=0.2*exp(pred1) + 0.2*exp(pred2) + 0.2*exp(pred3) + 0.2*exp(pred4) + 0.2*exp(pred5)

#Gbm training

train=read.csv("train.csv")
test=read.csv("test.csv")
train=subset(train,train$Hazard<=20)
train$Hazard=as.factor(train$Hazard)

gbm1= gbm(Hazard~., 
                    data=train,
                    distribution = "laplace",
                    n.trees = 500,
                    interaction.depth = 20,
                    n.minobsinnode = 500,
                    shrinkage = 0.05,
                    bag.fraction = 0.9,cv.folds=10)

pred_gbm=predict(gbm1,test[,-1],n.trees=500,type="response")

#xboost Modeling

X=read.csv("train.csv")
X.test=read.csv("test.csv")


# extract id
id.test = X.test$Id
X.test$Id = NULL
X$Id = NULL
n = nrow(X)

# extarct target
y = X$Hazard
X$Hazard = NULL

# replace factors with level mean hazard
for (i in 1:ncol(X)) {
  if (class(X[,i])=="factor") {
    mm <- aggregate(y~X[,i], data=X, mean)
    levels(X[,i]) <- as.numeric(mm[,2]) 
    levels(X.test[,i]) <- mm[,2] 
    X[,i] <- as.numeric(as.character(X[,i]))  
    X.test[,i] <- as.numeric(as.character(X.test[,i]))
  }
}
X = as.matrix(X)
X.test = as.matrix(X.test)

# train & tune --skipped--
logfile <- data.frame(shrinkage=c(0.04, 0.03, 0.03, 0.03, 0.02),
                      rounds = c(140, 160, 170, 140, 180),
                      depth = c(8, 7, 9, 10, 10),
                      gamma = c(0, 0, 0, 0, 0),
                      min.child = c(5, 5, 5, 5, 5),
                      colsample.bytree = c(0.7, 0.6, 0.65, 0.6, 0.85),
                      subsample = c(1, 0.9, 0.95, 1, 0.6))

# generate final prediction -- bag of 50 models --
models = 5
repeats = 10
yhat.test  = rep(0,nrow(X.test))
for (j in 1:repeats) {
  for (i in 1:models){
    set.seed(j*1000 + i*100)
    xgboost.mod = xgboost(data = X, label = y, max.depth = logfile$depth[i], eta = logfile$shrinkage[i],
                           nround = logfile$rounds[i], nthread = 4, objective = "reg:linear", subsample=logfile$subsample[i],
                           colsample_bytree=logfile$colsample.bytree[i], gamma=logfile$gamma[i], min.child.weight=logfile$min.child[i])
    yhat.test = yhat.test + predict(xgboost.mod, X.test)  
  }
}

pred_xboost =  yhat.test/(models*repeats)


final_pred=pred_xboost*.7+pred_gbm*.2+pred_random*.1


Submission=data.frame("Id"=id.test,"train$Hazard"=final_pred)
write.csv(Submission,"Submission.csv",row.names=FALSE,quote=FALSE)



