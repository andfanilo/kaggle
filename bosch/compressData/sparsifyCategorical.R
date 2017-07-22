library(data.table)
library(FeatureHashing)
library(Matrix)

hash <- function(x) ifelse(is.na(x), 0, hashed.value(x) %% 2^16)

batches   <- 20
col_first <- seq(2, 2140, by = 2140 / batches)
col_last  <- col_first + (2140 / batches) - 1

processed_data <- vector("list", length(col_first))

for(i in seq_along(col_first)) {
  
  dt <- fread("data/test_categorical.csv",
              colClasses = "character",
              na.strings = "",
              select = c(col_first[i]:col_last[i]))
  
  for(col in names(dt)) set(dt, j = col, value = hash(dt[[col]]))
  
  processed_data[[i]] <- Matrix(as.matrix(dt), sparse = T)
  
}

all_data <- do.call("cbind", processed_data)
saveRDS(all_data, file = "data/test_categorical.rds")