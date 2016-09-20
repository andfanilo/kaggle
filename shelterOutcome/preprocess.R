library(dplyr) # for data manip like bind_rows
library(lubridate) # for hour and date stuff
library(rpart) # rpart for regression trees for imputation
library(caret)
library(Matrix)

# Read the data
train <- read.csv('data/train.csv', stringsAsFactors = F)
test <- read.csv('data/test.csv', stringsAsFactors = F)

# Rename the ID column so train & test match
names(train)[1] <- 'ID'

# And change ID in test to character
test$ID <- as.character(test$ID)

# Combine test & training data
full <- bind_rows(train, test)

# factor(full$AgeuponOutcome)[1:10]

# Get the time value:
full$TimeValue <- sapply(full$AgeuponOutcome,  
                         function(x) strsplit(x, split = ' ')[[1]][1])

# Now get the unit of time:
full$UnitofTime <- sapply(full$AgeuponOutcome,  
                          function(x) strsplit(x, split = ' ')[[1]][2])

# Fortunately any "s" marks the plural, so we can just pull them all out
full$UnitofTime <- gsub('s', '', full$UnitofTime)

full$TimeValue  <- as.numeric(full$TimeValue)
full$UnitofTime <- as.factor(full$UnitofTime)

# Make a multiplier vector
multiplier <- ifelse(full$UnitofTime == 'day', 1,
              ifelse(full$UnitofTime == 'week', 7,
              ifelse(full$UnitofTime == 'month', 30, # Close enough
              ifelse(full$UnitofTime == 'year', 365, NA))))

# Apply our multiplier
full$AgeinDays <- full$TimeValue * multiplier

# summary(full$AgeinDays)

# Replace blank names with "Nameless"
full$Name <- ifelse(nchar(full$Name)==0, 'Nameless', full$Name)

# Make a name v. no name variable
full$HasName[full$Name == 'Nameless'] <- 0
full$HasName[full$Name != 'Nameless'] <- 1

# Replace blank sex with most common
full$SexuponOutcome <- ifelse(nchar(full$SexuponOutcome)==0, 
                              'Spayed Female', full$SexuponOutcome)

# Extract time variables from date (uses the "lubridate" package)
full$Hour    <- hour(full$DateTime)
full$Weekday <- wday(full$DateTime)
full$Month   <- month(full$DateTime)
full$Year    <- year(full$DateTime)

# Time of day may also be useful
full$TimeofDay <- ifelse(full$Hour > 5 & full$Hour < 11, 'morning',
                  ifelse(full$Hour > 10 & full$Hour < 16, 'midday',
                  ifelse(full$Hour > 15 & full$Hour < 20, 'lateday', 
                  'night')))

# Put factor levels into the order we want
full$TimeofDay <- factor(full$TimeofDay, 
                         levels = c('morning', 'midday',
                                    'lateday', 'night'))

# Use "grepl" to look for "Mix"
full$IsMix <- ifelse(grepl('Mix', full$Breed), 1, 0)

# Split on "/" and remove " Mix" to simplify Breed
full$SimpleBreed <- sapply(full$Breed, 
                           function(x) gsub(' Mix', '', 
                                            strsplit(x, split = '/')[[1]][1]))

# Use strsplit to grab the first color
full$SimpleColor <- sapply(full$Color, 
                           function(x) strsplit(x, split = '/| ')[[1]][1])

# Use "grepl" to look for "Intact"
full$Intact <- ifelse(grepl('Intact', full$SexuponOutcome), 1,
                      ifelse(grepl('Unknown', full$SexuponOutcome), 'Unknown', 0))

# Use "grepl" to look for sex
full$Sex <- ifelse(grepl('Male', full$SexuponOutcome), 'Male',
                   ifelse(grepl('Unknown', full$Sex), 'Unknown', 'Female'))

# Use rpart to predict the missing age values
age_fit <- rpart(AgeinDays ~ AnimalType + Sex + Intact + SimpleBreed + HasName, 
                 data = full[!is.na(full$AgeinDays), ], 
                 method = 'anova')

# Impute predicted age values where missing using "predict"
full$AgeinDays[is.na(full$AgeinDays)] <- predict(age_fit, full[is.na(full$AgeinDays), ])

# Use the age variable to make a puppy/kitten variable
full$Lifestage[full$AgeinDays < 365] <- 'baby'
full$Lifestage[full$AgeinDays >= 365] <- 'adult'

full$Lifestage <- factor(full$Lifestage)

# Factor columns should be defined
factorVars <- c('Name','OutcomeType','OutcomeSubtype','AnimalType',
                'SexuponOutcome','AgeuponOutcome','SimpleBreed','SimpleColor',
                'HasName','IsMix','Intact','Sex','TimeofDay','Lifestage')

full[factorVars] <- lapply(full[factorVars], function(x) as.factor(x))

# Define variables we will use in ML
features <- c("AnimalType", "AgeinDays", "HasName", "Hour", 
              "Weekday", "Month", "Year", "TimeofDay", 
              "IsMix", "Intact", "SimpleBreed", "SimpleColor", 
              "Intact", "Sex", "Lifestage")
label <- "OutcomeType"

x_train <- full[1:26729, features]
x_test <- full[26730:nrow(full), features]

y_train <- y_train <- as.numeric(as.factor(train$OutcomeType)) - 1
labels_train <- data.frame(train$OutcomeType, y_train)