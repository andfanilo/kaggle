library(bit64)
library(data.table)
library(Matrix)
library(caret)

# Set a random seed
set.seed(731)

train_numeric <- readRDS("data/train_numeric.rds")
test_numeric <- readRDS("data/test_numeric.rds")

train_categorical <- readRDS("data/train_categorical.rds")
test_categorical <- readRDS("data/test_categorical.rds")

sample_submission <- fread("data/sample_submission.csv")

y <- train_numeric[,970]
test_ids <- test_numeric[,1]

folds <- createFolds(as.factor(y), k = 6)
valid <- folds$Fold1
model <- c(1:length(y))[-valid]