#library(tsne)
#
#tsne_out <- tsne(train_numeric[1:3000,2:969], perplexity = 50)

library(Rtsne)

tsne_out <- Rtsne(train_numeric[1:300,2:969], perplexity = 50)
