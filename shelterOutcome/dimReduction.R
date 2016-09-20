library(FactoMineR)

# number of categories per variable
# cats = apply(x_train, 2, function(x) nlevels(as.factor(x)))

MCA(
  x_train,
  quanti.sup = c(2,4,5:7),
  quali.sup = c(1,3,8:15)
    )
