library(FactoMineR)
library(caret)

full_factor <- full
full_factor[, `:=`(people_id=NULL, activity_id=NULL)]

for (col in names(full_factor)) set(full_factor, j = col, value = as.factor(full_factor[[col]]))

#for (col in names(full_factor)) print(paste0("Column ", col, " has length ", length(unique(full[, get(col)]))))
### [1] "Column group_1 has length 34224"
### [1] "Column char_10.x has length 6970"

x_train <- full_factor[1:nrow(act_train), .(outcome, group_1)]
x_test <- full_factor[(nrow(act_train)+1):nrow(act_full), .(outcome, group_1)]

mca <- MCA(x_train[1:10000,], quali.sup = c(1))
#pca$eig # indicates 95% inertia in 25 first compo

