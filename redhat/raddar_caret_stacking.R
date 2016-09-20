library(data.table)
library(FeatureHashing)
library(xgboost)
library(dplyr)
library(Matrix)
library(SparseM)
library(irlba)
require(caret)

# Source : https://htmlpreview.github.io/?https://github.com/zachmayer/caretEnsemble/blob/master/inst/doc/caretEnsemble-intro.html

### Read data
train=fread('data/act_train.csv') #%>% as.data.frame()
test=fread('data/act_test.csv') #%>% as.data.frame()

### People data frame
people=fread('data/people.csv') #%>% as.data.frame()
people$char_1<-NULL #unnecessary duplicate to char_2
names(people)[2:length(names(people))]=paste0('people_',names(people)[2:length(names(people))])

## Transform logical cols to numbers
p_logi <- names(people)[which(sapply(people, is.logical))]
for (col in p_logi) set(people, j = col, value = as.numeric(people[[col]]))

## Reducing group_1 dimension
people$people_group_1[people$people_group_1 %in% names(which(table(people$people_group_1)==1))]='group unique'

## Reducing char_10 dimension
unique.char_10=
  rbind(
    select(train,people_id,char_10),
    select(train,people_id,char_10)) %>% group_by(char_10) %>% 
  summarize(n=n_distinct(people_id)) %>% 
  filter(n==1) %>% 
  select(char_10) %>%
  as.matrix() %>% 
  as.vector()

train$char_10[train$char_10 %in% unique.char_10]='type unique'
test$char_10[test$char_10 %in% unique.char_10]='type unique'

### Merge things
d1 <- merge(train, people, by = "people_id", all.x = T)
d2 <- merge(test, people, by = "people_id", all.x = T)
Y <- d1$outcome
d1$outcome <- NULL
row.train=nrow(train)

gc()

D=rbind(d1,d2)
D$i=1:dim(D)[1]

test_activity_id=test$activity_id
rm(train,test,d1,d2);gc()

char.cols=c('activity_category','people_group_1',
            'char_1','char_2','char_3','char_4',
            'char_5','char_6','char_7','char_8',
            'char_9','char_10', 'people_char_2','people_char_3',
            'people_char_4','people_char_5','people_char_6',
            'people_char_7','people_char_8','people_char_9')

for (f in char.cols) {
  if (class(D[[f]])=="character") {
    levels <- unique(c(D[[f]]))
    D[[f]] <- as.numeric(factor(D[[f]], levels=levels))
  }
}

D.sparse=
  cBind(sparseMatrix(D$i,D$activity_category),
        sparseMatrix(D$i,D$people_group_1),
        sparseMatrix(D$i,D$char_1),
        sparseMatrix(D$i,D$char_2),
        sparseMatrix(D$i,D$char_3),
        sparseMatrix(D$i,D$char_4),
        sparseMatrix(D$i,D$char_5),
        sparseMatrix(D$i,D$char_6),
        sparseMatrix(D$i,D$char_7),
        sparseMatrix(D$i,D$char_8),
        sparseMatrix(D$i,D$char_9),
        sparseMatrix(D$i,D$people_char_2),
        sparseMatrix(D$i,D$people_char_3),
        sparseMatrix(D$i,D$people_char_4),
        sparseMatrix(D$i,D$people_char_5),
        sparseMatrix(D$i,D$people_char_6),
        sparseMatrix(D$i,D$people_char_7),
        sparseMatrix(D$i,D$people_char_8),
        sparseMatrix(D$i,D$people_char_9)
  )

D.sparse=
  cBind(D.sparse,
        D$people_char_10,
        D$people_char_11,
        D$people_char_12,
        D$people_char_13,
        D$people_char_14,
        D$people_char_15,
        D$people_char_16,
        D$people_char_17,
        D$people_char_18,
        D$people_char_19,
        D$people_char_20,
        D$people_char_21,
        D$people_char_22,
        D$people_char_23,
        D$people_char_24,
        D$people_char_25,
        D$people_char_26,
        D$people_char_27,
        D$people_char_28,
        D$people_char_29,
        D$people_char_30,
        D$people_char_31,
        D$people_char_32,
        D$people_char_33,
        D$people_char_34,
        D$people_char_35,
        D$people_char_36,
        D$people_char_37,
        D$people_char_38,
        D$binay_sum)

# Fast PCA computing
pc <- irlba(D.sparse, nu=5)$u

train.sparse=pc[1:row.train,]
test.sparse=pc[(row.train+1):nrow(D.sparse),]

### Caret ---

set.seed(825)

fitControl <- trainControl(
  method = "repeatedcv",
  number = 2,
  repeats = 2
  )

gbmFit <- train(
  x = train.sparse, 
  y = as.factor(Y),
  method = "glm",
  family="binomial",
  trControl = fitControl,
  verbose = T
  )
