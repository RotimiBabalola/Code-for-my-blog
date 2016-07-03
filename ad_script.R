#Clear workspace
rm(list=ls())

#Load the dataset
ad_Data <- read.csv(file.choose()) #Browse to location of dataset on your computer


#HANDLING MISSING DATA
library(missForest)

#Let's use the missForest package to handle missing data --> uses Random Forest to predict missing values
ad_Data_imp <- missForest(ad_Data)

require(plyr)
require(dplyr)

ad_NoNA <- ad_Data_imp$ximp

#I want to create two columns with the "correct" values --- for aspect ratio and V4
#Correct the Aspect ratio column
ad_NoNA <- ad_NoNA %>%
                mutate(im_V3 = V2/V1)

#Round the V4 column since its can be either 0 or 1
ad_NoNA <- ad_NoNA %>%
                mutate(im_V4 = round(V4))

#Remove the V3 and V4 columns
ad_NoNA$V3 <- NULL
ad_NoNA$V4 <- NULL

#Re-order dataframe
ad_NoNA <-ad_NoNA[, c(1, 2, 1558, 1559, 3:1557)]

#Randomly order the data
set.seed(123)
ad_NoNA <- ad_NoNA[order(runif(3279)), ]

#Split into training(75%) and testing(25%)
train <- ad_NoNA[1:2459, ]
test <- ad_NoNA[2460:nrow(ad_Data), ]

#First, let's try a decision tree --- no boosting
library(C50)
treeModel <- C5.0(train[-1559], train$V1559)
treeModel_preds <- predict(treeModel, test)

#Confusion matrix
require(caret)
confusionMatrix(treeModel_preds, test$V1559, positive="ad.")

#Let's try boosting
boosted_treeModel <- C5.0(train[-1559], train$V1559, trials = 10)
boosted_preds <- predict(boosted_treeModel, test)

#Confusion matrix
confusionMatrix(boosted_preds, test$V1559, positive = "ad.")

#PARAMETER TUNING
ctrl <- trainControl(method = "cv", number = 5, selectionFunction = "oneSE")
grid <- expand.grid(.model = "tree", .trials = c(1, 5, 10, 15, 20, 25, 30, 35), 
                        .winnow = "FALSE")

best_model <- train(V1559 ~ ., data = train, method = "C5.0", metric = "Kappa", trControl = ctrl,
                    tuneGrid = grid)


best_model_preds <- predict(best_model, test)

#Confusion matrix
confusionMatrix(best_model_preds, test$V1559, positive = "ad.")


#LOGISTIC REGRESSION (REGULARIZED)
reg_model <- glmnet(as.matrix(train[-1559]), as.integer(train$V1559), family = "binomial")

#Make predictions
#This gives us all predictions for the entire lambda sequence used to create the model
reg_preds <- predict(reg_model, newx = as.matrix(test[-1559]), type = "response")

#Let's use lambda of 0.0001, 0.01 and 0.01 to make predictions
reg_preds_2 <- predict(reg_model, newx = as.matrix(test[-1559]), type = "response", s = c(0.0001, 0.001, 0.01))

#Convert the predictions matrix to a dataframe
reg_preds_2 <- as.data.frame(reg_preds_2)
#Change column names
names(reg_preds_2) <- c("lambda_0.0001", "lambda_0.001", "lambda_0.01")

#Create three columns that will contain the class predictions for each value of lambda
#I am using 0.5 as the probability threshold for classification
reg_preds_2 <- reg_preds_2 %>%
                mutate(pred_0.0001 = ifelse(lambda_0.0001 > 0.5, "nonad.", "ad."))

reg_preds_2 <- reg_preds_2 %>%
                mutate(pred_0.001 = ifelse(lambda_0.001 > 0.5, "nonad.", "ad."))

reg_preds_2 <- reg_preds_2 %>%
                mutate(pred_0.01 = ifelse(lambda_0.01 > 0.5, "nonad.", "ad."))

#Confusion matrix
confusionMatrix(reg_preds_2$pred_0.0001, test$V1559, positive = "ad.")

confusionMatrix(reg_preds_2$pred_0.001, test$V1559, positive = "ad.")

confusionMatrix(reg_preds_2$pred_0.01, test$V1559, positive = "ad.")

#Best model ---> lambda = 0.001









