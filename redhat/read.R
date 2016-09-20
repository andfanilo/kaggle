library(data.table)
library(dplyr)
library(lubridate)
library(Hmisc)

### Read all the data
act_train <- fread("data/act_train.csv", na.strings = "")
act_test <- fread("data/act_test.csv", na.strings = "")
people <- fread("data/people.csv", na.strings = "")

### Bind all activities and merge people info
### Preserve order of all train data on the top
act_full  <- bind_rows(act_train, act_test)
full <- left_join(act_full, people, by="people_id")
full <- data.table(full[order(match(full$activity_id, act_full$activity_id)),])

### Remove useless datasets
rm(people)

### Extract time variables from date for activity
full$weekday_x <- wday(full$date.x)
full$month_x   <- month(full$date.x)
full$year_x    <- year(full$date.x)
full$isWeekend_x <- as.numeric(full$weekday_x > 5)

#full$weekday_y <- wday(full$date.y)
#full$month_y   <- month(full$date.y)
#full$year_y    <- year(full$date.y)

### Lag between date.y and date.x in weeks
full$lag <- as.period(interval(full$date.y , full$date.x), unit="day") / as.period(dweeks(1))

### Rolling count number of activities per people id, order by date
full[, number_occurences := .N, by=people_id]
full$unit <- 1
full[order(date.x), cumulated_act := cumsum(unit), by=people_id]

### Impute NA cols with -1
na_cols <- setdiff(colnames(full)[!complete.cases(t(full))], "outcome")
for (col in na_cols) set(full, j = col, value = impute(full[[col]], fun = "-1")) # if want to impute

### Looks like char_1.x...char_9.x is NA when activity_category is not 1, 
### and char_10.x is NA when activity_category is 1, prove that

### TODO :  does char 38 vary per peoplid & date ?

### Remove dates now so we don't unecessarily encode them
full[, `:=`(date.x=NULL, date.y=NULL)] 

### Convert all booleans to numeric
logical_col <- names(full)[which(sapply(full, is.logical))]
for (col in logical_col) set(full, j = col, value = as.numeric(full[[col]]))

### Convert all strings to numeric levels
char_col <- names(full)[which(sapply(full, is.character))]
for (col in char_col) {
  if(col != "activity_id") {
      levels <- unique(c(full[[col]]))
      full[[col]] <- as.numeric(factor(full[[col]], levels=levels))
  }
}
#for (col in char_col)  print(paste0("Column ", col, " has length ", length(unique(full[, get(col)]))))

### Store IDs of train/test, and labels train
y_train <- full[1:nrow(act_train),]$outcome
act_id_train <- full[1:nrow(act_train), activity_id]
act_id_test <- full[(nrow(act_train)+1):nrow(act_full), activity_id]

### Convert necessary ones to string
numeric_cols <- c("outcome", "char_38", "lag", "number_occurences", "cumulated_act")
for (col in setdiff(names(full), numeric_cols)) set(full, j = col, value = as.character(full[[col]]))

### TODO : bucket all columns by count!

### I extracted features from both users and sessions files.
### In the case of users, I applied a one-hot-encoding representation for categorical features 
### and computed several features from age and dates, 
### using different techniques for dealing with missing values. 
### The sessions data was aggregated by user's id and different features based on counts, 
### frequency, unique values, etc. were computed. 

### Reorder columns to put outcome label first
setcolorder(full, c("outcome", setdiff(names(full), "outcome")))

### Delete label and id, don't need them in sparsifying matrix
### Destroy peson_id, maybe overfitting on it
full[, `:=`(people_id=NULL, unit=NULL)]
full[, `:=`(activity_id=NULL, outcome=NULL)] 