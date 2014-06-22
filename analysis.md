Predicting exercise type by using accelerometers 
============================================================

```
## Loading required package: lattice
## Loading required package: ggplot2
```



# Synopsis
The goal of the study is to be able to predict exercise type using readings from accelerometers.

This research uses data from [Human Activity Recognition project](http://groupware.les.inf.puc-rio.br/har)

# Data Processing
## Downloading data
At the start data is downloaded and extracted. Alternate way to obtain the data is to download it manually from [link1](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv) and  [link2](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv).

```r
require(R.utils)
# download training dataset
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", 
    method = "curl", destfile = "pml-training.csv")
# download testing dataset
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", 
    method = "curl", destfile = "pml-testing.csv")
```


## Loading data
Data is loaded using read.csv function:

```r
pml_training = read.csv("pml-training.csv", stringsAsFactors = FALSE, na.strings = c("", 
    "NA"))

pml_testing = read.csv("pml-testing.csv", stringsAsFactors = FALSE, na.strings = c("", 
    "NA"))
```


## Preprocessing data
Lets look at the data using str function: 

```r
str(pml_training)
```

There are unnecessary columns that are not accelometers readings nor classe column. Let's remove them:


```r
dropCol <- c("X", "user_name", "raw_timestamp_part_1", "raw_timestamp_part_2", 
    "cvtd_timestamp", "new_window", "num_window")
pml_training <- subset(pml_training, select = !names(pml_training) %in% dropCol)
pml_testing <- subset(pml_testing, select = !names(pml_training) %in% dropCol)
```


There are also a lot of columns with more than 75% of NAs values. We will not use this columns in our prediction model:

```r
classe <- pml_training$classe
pml_training$classe <- NULL

goodObs <- sapply(pml_training, function(x) sum(!is.na(x))/length(x) < 0.25)
n <- names(goodObs)[goodObs == FALSE]

tidy_pml_training <- pml_training[, n]
tidy_pml_testing <- pml_testing[, n]

tidy_pml_training$classe <- as.factor(classe)
```


To check our future assumptions of the out of sample error, we will divide our training data into train and test datasets:


```r
set.seed(123)
inTrain <- createDataPartition(y = tidy_pml_training$classe, p = 0.8, list = FALSE)
training <- tidy_pml_training[inTrain, ]
testing <- tidy_pml_training[-inTrain, ]
```


## Model creation
We use random forest classification with 10-fold cross validation to build a machine learning algorithm and make assumptions of the out of sample error.

```r
# using 10-fold CV
fitControl <- trainControl(method = "cv", number = 10, savePred = T, classProb = T)
RFfit <- train(classe ~ ., data = training, method = "rf", trControl = fitControl)
```

```
## Loading required package: randomForest
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
```

```r
RFfit
```

```
## Random Forest 
## 
## 15699 samples
##    52 predictors
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## 
## Summary of sample sizes: 14129, 14128, 14130, 14130, 14132, 14128, ... 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa  Accuracy SD  Kappa SD
##   2     1         1      0.003        0.004   
##   30    1         1      0.002        0.002   
##   50    1         1      0.003        0.004   
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
```

As we can see, we should have the out of sample error close to zero. Let's check our out of sample error for our testing dataset:

```r
# using 10-fold CV
testing$predicted <- predict(RFfit, newdata = testing)
accuracy <- sum(diag(table(testing$predicted, testing$classe)))/nrow(testing)
accuracy
```

```
## [1] 0.9939
```

Predicting testing dataset yielded ~0.99 accuracy!
Let's create confusion matrix:

```r
cm <- confusionMatrix(data = testing$predicted, testing$classe)
cm
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1115    8    0    0    0
##          B    1  749    4    0    0
##          C    0    2  677    4    0
##          D    0    0    3  638    1
##          E    0    0    0    1  720
## 
## Overall Statistics
##                                         
##                Accuracy : 0.994         
##                  95% CI : (0.991, 0.996)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.992         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.999    0.987    0.990    0.992    0.999
## Specificity             0.997    0.998    0.998    0.999    1.000
## Pos Pred Value          0.993    0.993    0.991    0.994    0.999
## Neg Pred Value          1.000    0.997    0.998    0.998    1.000
## Prevalence              0.284    0.193    0.174    0.164    0.184
## Detection Rate          0.284    0.191    0.173    0.163    0.184
## Detection Prevalence    0.286    0.192    0.174    0.164    0.184
## Balanced Accuracy       0.998    0.993    0.994    0.996    0.999
```

# Results
Now we can easily predict our pml_testing dataset classe values:

```r
answers <- predict(RFfit, newdata = pml_testing)
```

And write them to files:


```r
pml_write_files = function(x) {
    n = length(x)
    for (i in 1:n) {
        filename = paste0("problem_id_", i, ".txt")
        write.table(x[i], file = filename, quote = FALSE, row.names = FALSE, 
            col.names = FALSE)
    }
}

pml_write_files(answers)
```

