library(FeatureHashing)
library(xgboost)

# Set a random seed
set.seed(731)

# Store IDs of train/test, and labels train
y_train <- full[1:nrow(act_train),]$outcome
act_id_train <- full[1:nrow(act_train), activity_id]
act_id_test <- full[(nrow(act_train)+1):nrow(act_full), activity_id]

# Delete label and id, don't need them in sparsifying matrix
full[, `:=`(activity_id=NULL, outcome=NULL)]

# All to string
for (col in names(full)) set(full, j = col, value = as.character(full[[col]]))

# Char 38 is numeric
full$char_38 <- as.numeric(full$char_38)

# One hot encoding by hashing trick
b <- hash.size(full)
full.oneHotEncoded <- hashed.model.matrix(~.-1, full, b, create.mapping = T)

xgb_train <- xgb.DMatrix(data = full.oneHotEncoded[1:nrow(act_train),], label=y_train)
xgb_test <- xgb.DMatrix(data = full.oneHotEncoded[(nrow(act_train)+1):nrow(act_full),])

## quick xgboost
bst <- xgboost(xgb_train, 
               y_train, 
               nrounds=100,
               eval_metric = "auc",
               objective='binary:logistic',
               max_depth=6
)

predictions <- predict(bst, xgb_test)
xgb_preds <- data.frame(act_id_test, predictions)
names(xgb_preds) <- c('activity_id','outcome')

write.csv(xgb_preds, 'data/submission.csv', row.names = FALSE)
