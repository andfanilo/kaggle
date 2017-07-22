valid_preds <- (valid_preds_cat + valid_preds_num) / 2
matt <- data.table(thresh = seq(0.990, 0.999, by = 0.001))
matt$scores <- sapply(matt$thresh, FUN =
                        function(x) mc(y[valid], (valid_preds > quantile(valid_preds, x)) * 1))
best_thresh <- matt$thresh[which(matt$scores == max(matt$scores))][1]

test_preds <- (test_preds_cat + test_preds_num) / 2
options(scipen = 999) # remove exponential form from file -_-
sub   <- data.table(Id = as.numeric(test_ids),
                    Response = (test_preds > quantile(test_preds, best_thresh)) * 1,
                    Probs = test_preds)

write.csv(sub[, .(Id, Response)], "submission/quickAverageXGBoost.csv", row.names = F)
write.csv(sub[, .(Id, Probs)], "submission/quickAverageXGBoost_probs.csv", row.names = F)

# recooooord !!!