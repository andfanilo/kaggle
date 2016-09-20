library(randomForest)

# build model - i like verbose output ;-)
# sad panda, can't put simpleBreed in it easily
rf <- randomForest(OutcomeType~AnimalType+AgeinDays+HasName+
                     Hour+Weekday+Month+Year+TimeofDay+
                     Intact+IsMix+SimpleColor+Sex+Lifestage, data=full[1:26729, ],
                   importance=FALSE, do.trace=1, ntree=550)

rf_preds <- data.frame(predict(rf, full[26730:nrow(full), ], type='vote'))

# buuuut I like SimpleBreed, try again with one hot encoded from xgboost script
# actually it worsens the score XD
#rf <- randomForest(x = x_train.oneHotEncoded,
#                   y = as.factor(train$OutcomeType),
#                   importance=FALSE, 
#                   do.trace=1, ntree=550
#                   )
#rf_preds <- data.frame(predict(rf, x_test.oneHotEncoded, type='vote'))

# take a look
head(rf_preds)