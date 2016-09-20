# Source : https://www.kaggle.com/laurae2/predicting-red-hat-business-value/raddar-goes-ham-with-leak
library(data.table)

loiso <- fread("data/loiso_submission.csv")
keras <- fread("data/submission_residual_32_64.csv")
raddar <- fread("data/raddar_submission.csv")

t <- merge(raddar, keras, by = "activity_id")
t$outcome <- (t$outcome.x + t$outcome.y) / 2
t$outcome.x <- NULL
t$outcome.y <- NULL

use_raddar <- merge(loiso[is.na(loiso$filled), ], t, by = "activity_id", all.x = T)
use_loiso <- merge(loiso, use_raddar, by = "activity_id", all.x = T)

use_loiso$filled.x[is.na(use_loiso$filled.x)] <- use_loiso$outcome[is.na(use_loiso$filled.x)]
predictions <- data.frame(activity_id = use_loiso$activity_id, outcome = use_loiso$filled.x, stringsAsFactors = FALSE)

write.csv(predictions, file="data/merge_loiso_raddar_keras_submission.csv", row.names=FALSE)