# Average the raddar's + Loiso leak merge

filenames <- list.files("raddar_grid", pattern="*.csv", full.names=TRUE)

ldf <- lapply(filenames, fread)
act_id <- ldf[[1]]$activity_id
outcomes <- lapply(ldf, function(x) x$outcome)

sum <- outcomes[[1]]

for (i in 2:length(outcomes)) 
  sum = sum + outcomes[[i]]

raddar_mean <- sum / length(outcomes)
raddar <- data.frame(activity_id = act_id, outcome = raddar_mean)
loiso <- fread("data/loiso_submission.csv")

use_raddar <- merge(loiso[is.na(loiso$filled), ], raddar, by = "activity_id", all.x = T)
use_loiso <- merge(loiso, use_raddar, by = "activity_id", all.x = T)

use_loiso$filled.x[is.na(use_loiso$filled.x)] <- use_loiso$outcome[is.na(use_loiso$filled.x)]
predictions <- data.frame(activity_id = use_loiso$activity_id, outcome = use_loiso$filled.x, stringsAsFactors = FALSE)

write.csv(predictions, file="data/merge_loiso_raddar_grid_submission.csv", row.names=FALSE)