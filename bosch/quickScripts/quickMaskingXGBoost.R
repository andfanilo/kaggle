t <- which(train_categorical!=0,arr.ind = T)
mask <- sparseMatrix(i = t[,1], j = t[,2], x = 1)
rm(t)

train_numeric <- train_numeric[,2:969]
t <- train_numeric * mask

train_categorical[which(train_categorical!=0,arr.ind = T)] <- 1



mt <- Matrix(0, nrow = nrow(train_numeric), ncol = ncol(train_numeric), sparse = TRUE)

apply(t, 1, function(x)  mt[x[[1]], x[[2]]] <- train_numeric[x[[1]], x[[2]]])
rm(t)
train_numeric <- mt
rm(mt)