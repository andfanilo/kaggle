ave_pred <- xgb_preds

# drop ID column for averaging
ave_pred <- ave_pred[,1:5]
ave_pred %>% head()

# average predictions
ave_pred <- 0.5*(ave_pred+rf_preds)
ave_pred %>% head()

ave_pred['ID'] <- test['ID']

# write the submission file
write.csv(ave_pred[, c('ID', 
                       'Adoption', 
                       'Died', 
                       'Euthanasia', 
                       'Return_to_owner', 
                       'Transfer')], 
          'data/submission.csv', 
          row.names = FALSE)