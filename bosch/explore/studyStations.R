library(data.table)

t <- fread("data/train_numeric.csv", select=2:969, nrows = 1000)

# we have 51 stations. each have a number of features.
# we could eventually build a model per station XD 
# also could it be that when thing is broken, should not go down the whole pipeline ?

