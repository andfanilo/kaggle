library(bit64) # needed for integer IDs..?
library(data.table)

# Change font
#Sys.setlocale(category = "LC_ALL", locale = "chinese")
#windowsFonts(simkai=windowsFont("simkai"))

# Read the data
phone_brand_device_model <- fread("data/phone_brand_device_model.csv",
                                  colClasses=c("character","factor","factor"))
label_train <- fread("data/gender_age_train.csv", 
                     colClasses=c("character","factor","integer","factor"))
label_test  <- fread("data/gender_age_test.csv",colClasses=c("character"))

# Build train dataframe, so you keep ids linked to label
y_train <- as.numeric(as.factor(label_train$group)) - 1
labels_train <- data.frame(label_train$group, y_train)
num_class <- max(y_train)+1

# Get header for submissions
sample_submission <- fread("data/sample_submission.csv")
submission_header <- colnames(sample_submission)

# Advanced, more complex data
app_labels <- fread("data/app_labels.csv")
label_categories <- fread("data/label_categories.csv")

app_events <- fread("data/app_events.csv")
events <- fread("data/events.csv")
