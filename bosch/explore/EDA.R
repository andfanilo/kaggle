library(data.table)
library(ggplot2)
library(mice)

# Study Response variable ----
#table(train_numeric$Response)
#c <- ggplot(train_numeric, aes(factor(Response)))
#c + geom_bar()

# Percent of empty cells ----
#sum(is.na(train_numeric)) / (nrow(train_numeric)*ncol(train_numeric))

# Filter on specific columns ----
#t <- fread(input = "data/train_numeric.csv",
#      select = c(2,970),
#      header = TRUE,
#      sep = ",",
#      stringsAsFactors = FALSE)
#
#summary(t[Response==1, ][[1]])

# Missing values & unique paths ----
t <- fread("data/train_numeric.csv", select=2:969, nrows = 1000)
md.pattern(t)