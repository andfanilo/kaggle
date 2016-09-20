library(FeatureHashing)
library(Hmisc)

# Set a random seed
set.seed(731)

# Store IDs of train/test, and labels train
y_train <- full[1:nrow(act_train),]$outcome
act_id_train <- full[1:nrow(act_train), activity_id]
act_id_test <- full[(nrow(act_train)+1):nrow(act_full), activity_id]

# Delete label and id, don't need them in sparsifying matrix
full[, `:=`(activity_id=NULL, outcome=NULL)]

# Binning group_1 and char_10.x
for (col in names(full)) set(full, j = col, value = as.character(full[[col]]))

# Regroup by number of appearances
full[ , `:=`( count_group_1 = .N ) , by = group_1 ]
full[ , `:=`( count_char_10 = .N ) , by = char_10.x ]

full$group_1_bin <- with(full, cut2(full$count_group_1, g=10))
full$char_10_bin <- with(full, cut2(full$count_char_10, g=10))

full[, `:=`(count_group_1=NULL, count_char_10=NULL)]
full[, `:=`(group_1=NULL, char_10.x=NULL)] # destroy hours because only 1 value

# Char 38 is numeric
full$char_38 <- as.numeric(full$char_38)

# One hot encoding
full.oneHotEncoded <- sparse.model.matrix(~.-1, data = full)

xgb_train <- xgb.DMatrix(data = full.oneHotEncoded[1:nrow(act_train),], label=y_train)
xgb_test <- xgb.DMatrix(data = full.oneHotEncoded[(nrow(act_train)+1):nrow(act_full),])

#param <- list(objective = "binary:logistic", 
#eval_metric = "auc",
#booster = "gblinear", 
#eta = 0.02,
#subsample = 0.7,
#colsample_bytree = 0.7,
#min_child_weight = 0,
#max_depth = 10)

## quick xgboost
bst <- xgboost(xgb_train, 
               y_train, 
               nrounds=100,
               eval_metric = "auc",
               objective='binary:logistic',
               max_depth=6
)

# make predictions
predictions <- predict(bst, xgb_test)
xgb_preds <- data.frame(act_id_test, predictions)
names(xgb_preds) <- c('activity_id','outcome')

write.csv(xgb_preds, 'data/submission.csv', row.names = FALSE)