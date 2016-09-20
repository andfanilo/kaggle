# Source : https://www.kaggle.com/laurae2/predicting-red-hat-business-value/raddar-goes-ham-with-leak

loiso <- fread("data/loiso_submission.csv")
raddar <- fread("data/raddar_submission.csv")

use_raddar <- merge(loiso[is.na(loiso$filled), ], raddar, by = "activity_id", all.x = T)
use_loiso <- merge(loiso, use_raddar, by = "activity_id", all.x = T)

use_loiso$filled.x[is.na(use_loiso$filled.x)] <- use_loiso$outcome[is.na(use_loiso$filled.x)]
predictions <- data.frame(activity_id = use_loiso$activity_id, outcome = use_loiso$filled.x, stringsAsFactors = FALSE)

write.csv(predictions, file="data/merge_loiso_raddar_submission.csv", row.names=FALSE)