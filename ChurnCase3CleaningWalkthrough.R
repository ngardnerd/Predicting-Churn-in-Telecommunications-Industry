#An example of cleaning the churn data (works off of Case3-ChurnPredict.RData)

load("Case3-ChurnPredict.RData")

#If you want to use the data as-is, the following two lines give you that
TRAIN.ASIS <- CHURN[which(!is.na(CHURN$Churn)),]
TEST.ASIS <- CHURN[which(is.na(CHURN$Churn)),]




CHURN.ORIG <- CHURN  #Make a copy of CHURN in case we make a mistake.  If we mess up a column; CHURN[,i] <- CHURN.ORIG[,i]

names(CHURN)[2]
summary(CHURN[,2]); hist(CHURN[,2]); hist( log10( CHURN[,2] + 7.17 ) )  
#7.17 is arbitrary.  minimum is -6.17, so adding 7.17 makes the minimum value 1, so minimum of the log is 0
CHURN[,2] <- log10( CHURN.ORIG[,2] + 7.17 )

names(CHURN)[3]
summary(CHURN[,3]); hist(CHURN[,3]); hist( log10( CHURN[,3] + 1 ) )
# +1 is to make it so minimum value 1 (currently some have values of 0), so minimum of the log is 0

CHURN[,3] <- log10( CHURN.ORIG[,3] + 1 )

names(CHURN)[4]
summary(CHURN[,4]); hist(CHURN[,4]); hist( log10( CHURN[,4] + 13 ) )  #Probably ok as is

names(CHURN)[5]
summary(CHURN[,5]); hist(CHURN[,5]); hist( log10( CHURN[,5] + .1 ) )
# +0.1 is to make it so minimum value 0.1 (currently some have values of 0), so minimum of the log is -1
CHURN[,5] <- log10( CHURN.ORIG[,5] + 1 )

names(CHURN)[6]
summary(CHURN[,6]); hist(CHURN[,6]); hist( log10( CHURN[,6] + 1 ) )
# +1 is to make it so minimum value 1 (currently some have values of 0), so minimum of the log is 0
CHURN[,6] <- log10( CHURN.ORIG[,6] + 1 )

names(CHURN)[7]
summary(CHURN[,7]); hist(CHURN[,7]); hist( log10( CHURN[,7] + .1 ) )
# +1 is to make it so minimum value 1 (currently some have values of 0), so minimum of the log is 0
CHURN[,7] <- log10( CHURN.ORIG[,7] + 1 )

names(CHURN)[8]
summary(CHURN[,8]); hist(CHURN[,8]); hist( log10( CHURN[,8] + .1 ) )
# +0.1 is to make it so minimum value 0.1 (currently some have values of 0), so minimum of the log is -1
CHURN[,8] <- log10( CHURN.ORIG[,8] + 1 )

names(CHURN)[9]
summary(CHURN[,9]); hist(CHURN[,9]); hist( log10( CHURN[,9] + 1 ) )
# +1 is to make it so minimum value 1 (currently some have values of 0), so minimum of the log is 0
#This is a weird distribution; I wouldn't be against making it a factor with "None", "Some", "Many"
CHURN[,9] <- log10( CHURN.ORIG[,9] + 1 )

names(CHURN)[10]
summary(CHURN[,10]); hist(CHURN[,10]); hist( log10( CHURN[,10] + 1 ) )
# +1 is to make it so minimum value 1 (currently some have values of 0), so minimum of the log is 0
CHURN[,10] <- log10( CHURN.ORIG[,10] + 1 )

names(CHURN)[11]
summary(CHURN[,11]); hist(CHURN[,11]); hist( log10( CHURN[,11] + 1 ) )
# +1 is to make it so minimum value 1 (currently some have values of 0), so minimum of the log is 0
#This is a weird distribution; I wouldn't be against making it a factor with "None", "Some", "Many"
CHURN[,11] <- log10( CHURN.ORIG[,11] + 1 )

names(CHURN)[12]
summary(CHURN[,12]); hist(CHURN[,12]); hist( log10( CHURN[,12] + 1 ) )
# +1 is to make it so minimum value 1 (currently some have values of 0), so minimum of the log is 0
#This is a weird distribution; I wouldn't be against making it a factor with "None", "Some", "Many"
CHURN[,12] <- log10( CHURN.ORIG[,12] + 1 )

names(CHURN)[13]
summary(CHURN[,13]); hist(CHURN[,13]); hist( log10( CHURN[,13] + 1 ) )
# +1 is to make it so minimum value 1 (currently some have values of 0), so minimum of the log is 0
CHURN[,13] <- log10( CHURN.ORIG[,13] + 1 )

names(CHURN)[14]
summary(CHURN[,14]); hist(CHURN[,14]); hist( log10( CHURN[,14] + 1 ) )
# +1 is to make it so minimum value 1 (currently some have values of 0), so minimum of the log is 0
CHURN[,14] <- log10( CHURN.ORIG[,14] + 1 )

names(CHURN)[15]
summary(CHURN[,15]); hist(CHURN[,15]); hist( log10( CHURN[,15] + 1 ) )
# +1 is to make it so minimum value 1 (currently some have values of 0), so minimum of the log is 0
CHURN[,15] <- log10( CHURN.ORIG[,15] + 1 )

names(CHURN)[16]
summary(CHURN[,16]); hist(CHURN[,16]); hist( log10( CHURN[,16] + 1 ) )
# +1 is to make it so minimum value 1 (currently some have values of 0), so minimum of the log is 0
CHURN[,16] <- log10( CHURN.ORIG[,16] + 1 )

names(CHURN)[17]
summary(CHURN[,17]); hist(CHURN[,17]); hist( log10( CHURN[,17] + 1 ) )
# +1 is to make it so minimum value 1 (currently some have values of 0), so minimum of the log is 0
CHURN[,17] <- log10( CHURN.ORIG[,17] + 1 )

names(CHURN)[18]
summary(CHURN[,18]); hist(CHURN[,18]); hist( log10( CHURN[,18] + 1 ) )
# +1 is to make it so minimum value 1 (currently some have values of 0), so minimum of the log is 0
CHURN[,18] <- log10( CHURN.ORIG[,18] + 1 )

names(CHURN)[19]
summary(CHURN[,19]); hist(CHURN[,19]); hist( log10( CHURN[,19] + 1 ) )
table(CHURN[,19])
#With so few values, I felt it make sense to make this a categorical variable with two levels
CHURN[,19] <- factor( ifelse( CHURN.ORIG[,19]>0, "NonZero","Zero") )

names(CHURN)[20]
summary(CHURN[,20]); hist(CHURN[,20]); hist( log10( CHURN[,20] + .1 ) )
# +0.1 is to make it so minimum value 0.1 (currently some have values of 0), so minimum of the log is -1
# Might make sense to make this a 3 level categorical variable with none, some, many
CHURN[,20] <- log10( CHURN.ORIG[,20] + .1 )

names(CHURN)[21]
summary(CHURN[,21]); hist(CHURN[,21])
#Actually looks fine as-is!

names(CHURN)[22]
summary(CHURN[,22]); hist(CHURN[,22])
table(CHURN[,22])
#Not many unique values; let's group them into categories
replacement <- c()
for (i in 1:nrow(CHURN)) {
  if(CHURN.ORIG[i,22] == 1 ) { replacement[i] <- "One" }
  if(CHURN.ORIG[i,22] == 2 ) { replacement[i] <- "Two" }
  if(CHURN.ORIG[i,22] == 3 ) { replacement[i] <- "Three" }
  if(CHURN.ORIG[i,22] > 3 ) { replacement[i] <- "FourOrMore" }
}
CHURN[,22] <- factor(replacement)

names(CHURN)[23]
summary(CHURN[,23]); hist(CHURN[,23])
table(CHURN[,23])
#Not many unique values; let's group them into categories
replacement <- c()
for (i in 1:nrow(CHURN)) {
  if(CHURN.ORIG[i,23] <= 1 ) { replacement[i] <- "AtMostOne" }
  if(CHURN.ORIG[i,23] == 2 ) { replacement[i] <- "Two" }
  if(CHURN.ORIG[i,23] >= 3 ) { replacement[i] <- "ThreeOrMore" }
}
CHURN[,23] <- factor(replacement)

names(CHURN)[24]
summary(CHURN[,24]); hist(CHURN[,24]); hist( log10( CHURN[,24] + 1 ) )
table(CHURN[,24])
#Not many unique values; let's group them into categories
replacement <- c()
for (i in 1:nrow(CHURN)) {
  if(CHURN.ORIG[i,24] == 1 ) { replacement[i] <- "One" }
  if(CHURN.ORIG[i,24] == 2 ) { replacement[i] <- "Two" }
  if(CHURN.ORIG[i,24] == 3 ) { replacement[i] <- "Three" }
  if(CHURN.ORIG[i,24] >= 4 ) { replacement[i] <- "FourOrMore" }
}
CHURN[,24] <- factor(replacement)

names(CHURN)[25]
summary(CHURN[,25]); hist(CHURN[,25])
table(CHURN[,25])
#Not many unique values; let's group them into categories
replacement <- c()
for (i in 1:nrow(CHURN)) {
  if(CHURN.ORIG[i,25] == 1 ) { replacement[i] <- "One" }
  if(CHURN.ORIG[i,25] == 2 ) { replacement[i] <- "Two" }
  if(CHURN.ORIG[i,25] == 3 ) { replacement[i] <- "Three" }
  if(CHURN.ORIG[i,25] >= 4 ) { replacement[i] <- "FourOrMore" }
}
CHURN[,25] <- factor(replacement)

names(CHURN)[26]
summary(CHURN[,26]); hist(CHURN[,26])
#Actually seems fine as-is

names(CHURN)[27]
summary(CHURN[,27]); hist(CHURN[,27]) #odd because of zeroes
#0 is not a valid age.  But don't want to remove them.  Perhaps make categories?
table(CHURN[,27])
replacement <- c()
for (i in 1:nrow(CHURN)) {
  if(CHURN.ORIG[i,27] == 0 ) { replacement[i] <- "Unknown" }
  if(CHURN.ORIG[i,27] %in% 18:29 ) { replacement[i] <- "Youth" }
  if(CHURN.ORIG[i,27] %in% 30:45 ) { replacement[i] <- "Adult" }
  if(CHURN.ORIG[i,27] %in% 46:65 ) { replacement[i] <- "MiddleAged" }
  if(CHURN.ORIG[i,27] >= 66 ) { replacement[i] <- "Retired" }
}
CHURN[,27] <- factor( replacement )

names(CHURN)[28]
summary(CHURN[,28]); hist(CHURN[,28])  #same issue as AgeHH1
table(CHURN[,28])
replacement <- c()
for (i in 1:nrow(CHURN)) {
  if(CHURN.ORIG[i,28] == 0 ) { replacement[i] <- "Unknown" }
  if(CHURN.ORIG[i,28] %in% 18:29 ) { replacement[i] <- "Youth" }
  if(CHURN.ORIG[i,28] %in% 30:45 ) { replacement[i] <- "Adult" }
  if(CHURN.ORIG[i,28] %in% 46:65 ) { replacement[i] <- "MiddleAged" }
  if(CHURN.ORIG[i,28] >= 66 ) { replacement[i] <- "Retired" }
}
CHURN[,28] <- factor( replacement )


#These are all two-level categorical variables which are fine
#We'd only worry if there was a level that was exceedingly rare
names(CHURN)[29:42]
table(CHURN[,29])
table(CHURN[,30])
table(CHURN[,31])
table(CHURN[,32])
table(CHURN[,33])
table(CHURN[,34])
table(CHURN[,35])
table(CHURN[,36])
table(CHURN[,37])
table(CHURN[,38])
table(CHURN[,39])
table(CHURN[,40])
table(CHURN[,41])
table(CHURN[,42])


names(CHURN)[43]
table(CHURN[,43])
#Not many unique values; let's group them into categories.  One idea:
replacement <- c()
for (i in 1:nrow(CHURN)) {
  if(CHURN.ORIG[i,43] == 0 ) { replacement[i] <- "None" }
  if(CHURN.ORIG[i,43] == 1 ) { replacement[i] <- "One" }
  if(CHURN.ORIG[i,43] >= 2 ) { replacement[i] <- "TwoOrMore" }
}
CHURN[,43] <- factor(replacement)


names(CHURN)[44]
table(CHURN[,44])  #I think it makes sense to leave income as an integer instead of a category

names(CHURN)[45]
table(CHURN[,45])  #fine

names(CHURN)[46]
table(CHURN[,46])
#Not many unique values; let's group them into categories.  One idea:
replacement <- c()
for (i in 1:nrow(CHURN)) {
  if(CHURN.ORIG[i,46] == 0 ) { replacement[i] <- "None" }
  if(CHURN.ORIG[i,46] == 1 ) { replacement[i] <- "One" }
  if(CHURN.ORIG[i,46] >= 2 ) { replacement[i] <- "TwoOrMore" }
}
CHURN[,46] <- factor(replacement)


names(CHURN)[47]
table(CHURN[,47])  #Messy;  Convert into numbers first, then come up with categories.  One suggestion:
price <- as.numeric( as.character( CHURN.ORIG[,47] ) )
#However, this makes some values NA, so we still need to deal with them
replacement <- c()
for (i in 1:nrow(CHURN)) {
  if(is.na(price[i])) { next }
  if(price[i] <= 50 ) { replacement[i] <- "50OrLess" }
  if(price[i] > 50 & price[i] <= 100 ) { replacement[i] <- "50to100" }
  if(price[i] > 100 & price[i] < 200 ) { replacement[i] <- "100to200" }
  if(price[i] >= 200 ) { replacement[i] <- "200OrMore" }
}
#Make a categorical variable
replacement <- factor(replacement)
#Deal with missing values by adding a new level called Unknown and 
levels(replacement) <- c( levels(replacement), "Unknown" )
replacement[which(is.na(replacement))] <- "Unknown"
table(replacement)
CHURN[,47] <- replacement

names(CHURN)[48:51]  #These are all healthy variables
table(CHURN[,48])
table(CHURN[,49])
table(CHURN[,50])
table(CHURN[,51])


summary(CHURN)  #Scan;  better not have any NAs or "weird" values anymore
mean(complete.cases(CHURN[,-1]))  #Everything but Churn column is complete

#Split back into TRAIN/TEST
TRAIN.CLEAN <- CHURN[which(!is.na(CHURN$Churn)),]
TEST.CLEAN <- CHURN[which(is.na(CHURN$Churn)),]




##########################################################################
##########################################################################

# Modeling Building Time

##########################################################################
##########################################################################


### training models will take a loooooong time if we used all the data. 
### I split the data into training and holdout data to fix this


#TRAIN.CLEAN should be TRAIN.ASIS if you're not using the cleaned version
#Feel free to change the random number seed here so you're working on a slightly different version than your friends
set.seed(420); train.rows <- sample(1:nrow(TRAIN.CLEAN),10000)  
TRAIN <- TRAIN.CLEAN[train.rows,] 
HOLDOUT <- TRAIN.CLEAN[-train.rows,]  


##Set up how generalization error is estimated with caret.  Want AUC here since want to pick out the 
#customers with highest probabilities of churning (even if they are much less than 50%).
library(caret)
library(pROC)
fitControl <- trainControl(method="cv",number=5, classProbs=TRUE, 
                           summaryFunction=twoClassSummary, allowParallel = TRUE) 


#If parallelization is desired
library(doParallel)
library(parallel)

#Turn on
cluster <- makeCluster(detectCores() - 1) 
registerDoParallel(cluster) 

#After parallelization is no longer required, turn off:
#stopCluster(cluster) 
#registerDoSEQ() 



#Fit models and see what's best!  This example is for regularized logistic regression and vanilla partition.
#The postResample will get you accuracy on HOLDOUT, and roc() will get AUC on the HOLDOUT
#You'll be sad at the performance of the models you try out (churn is hard to predict), but you'll get
#a hint over what is working and not working
#Also, you'll find that the ACCURACY of your models won't be much better than the naive model.  That's fine.
#We're looking for a model with a high AUC.

glmnetGrid <- expand.grid(alpha = seq(0,1,.05),lambda = 10^seq(-4,-1,length=10))   

#Feel free to change the random number seed here so you're working on a slightly different version than your friends
set.seed(479); GLMnet <- train(Churn~.,data=TRAIN,method='glmnet', trControl=fitControl, tuneGrid=glmnetGrid,
                               preProc = c("center", "scale"))

#Ok to see the following, since indeed we want the area under the ROC curve and not accuracy, this is a good thing!
#Warning message:
#In train.default(x, y, weights = w, ...) :
#  The metric "Accuracy" was not in the result set. ROC will be used instead.

GLMnet$results[rownames(GLMnet$bestTune),]  #Just the row with the optimal choice of tuning parameter
postResample(predict(GLMnet,newdata=HOLDOUT),HOLDOUT$Churn)  #Error on holdout sample for reference
mean(HOLDOUT$Churn=="No")  #Accuracy of naive model that predicts everyone doesn't churn

roc(HOLDOUT$Churn,predict(GLMnet,newdata=HOLDOUT,type="prob")[,2])


#To glimpse how many of the "top" predictions are correct:
GLMnet.predictions <- predict(GLMnet,newdata=HOLDOUT,type="prob")[,2]
sum(GLMnet.predictions>=0.5)  #Number of individuals with predicted probabilities greater than 0.5 (a lift of 1.75)
cutoff <- sort(GLMnet.predictions,decreasing = TRUE)[2250]  #2250th largest predicted probability;  how many of those are right?
singled.out <- which(GLMnet.predictions >= cutoff)
length(singled.out)  #if not 2250, it will pick 2250 at random
singled.out <- sample(singled.out,size=2250) 
table( HOLDOUT$Churn[singled.out] )




#boosted Tree Model
paramGrid <- expand.grid(n.trees=3000,interaction.depth=5,shrinkage=.00015,n.minobsinnode=5)

set.seed(seed)


set.seed(420); GBM <- train(Churn~.,data=TRAIN,method='gbm',trControl=fitControl, tuneGrid = paramGrid, 
             preProc = c("center", "scale"),verbose=FALSE)

GBM$results[rownames(GBM$bestTune),]  #Just the row with the optimal choice of tuning parameter
postResample(predict(GBM,newdata=HOLDOUT),HOLDOUT$Churn)  #Error on holdout sample for reference
mean(HOLDOUT$Churn=="No")  #Accuracy of naive model that predicts everyone doesn't churn

roc(HOLDOUT$Churn,predict(GBM,newdata=HOLDOUT,type="prob")[,2])

GBM.predictions <- predict(GBM,newdata=HOLDOUT,type="prob")[,2]
sum(GBM.predictions>=0.5)  #Number of individuals with predicted probabilities greater than 0.5 (a lift of 1.75)
cutoff <- sort(GBM.predictions,decreasing = TRUE)[2250]  #2250th largest predicted probability;  how many of those are right?
singled.out <- which(GBM.predictions >= cutoff)
length(singled.out)  #if not 2250, it will pick 2250 at random
singled.out <- sample(singled.out,size=2250) 
table( HOLDOUT$Churn[singled.out] )

knnGrid <- expand.grid(k=1:50)   
KNNMODEL <- train(Churn~.,data=TRAIN,method='knn',tuneGrid=knnGrid,
                  trControl=fitControl, preProc = c("center", "scale"))


roc(HOLDOUT$Churn,predict(KNNMODEL,newdata=HOLDOUT,type="prob")[,2]) ### Performed terribly


#boosted tree 2
paramGrid2 <- expand.grid(n.trees=c(500,1000,1500),interaction.depth=5:6,shrinkage=c(.001,.0001),n.minobsinnode=7)

set.seed(seed)


set.seed(420); GBM2 <- train(Churn~.,data=TRAIN,method='gbm',trControl=fitControl, tuneGrid = paramGrid2, 
                            preProc = c("center", "scale"),verbose=FALSE)

GBM2$results[rownames(GBM2$bestTune),]  #Just the row with the optimal choice of tuning parameter
postResample(predict(GBM2,newdata=HOLDOUT),HOLDOUT$Churn)  #Error on holdout sample for reference
mean(HOLDOUT$Churn=="No")  #Accuracy of naive model that predicts everyone doesn't churn

roc(HOLDOUT$Churn,predict(GBM2,newdata=HOLDOUT,type="prob")[,2])

GBM2.predictions <- predict(GBM2,newdata=HOLDOUT,type="prob")[,2]
sum(GBM2.predictions>=0.5)  #Number of individuals with predicted probabilities greater than 0.5 (a lift of 1.75)
cutoff <- sort(GBM2.predictions,decreasing = TRUE)[2250]  #2250th largest predicted probability;  how many of those are right?
singled.out <- which(GBM2.predictions >= cutoff)
length(singled.out)  #if not 2250, it will pick 2250 at random
singled.out <- sample(singled.out,size=2250) 
table( HOLDOUT$Churn[singled.out] )


nnetGrid <- expand.grid(size=1:4,decay=10^( seq(-5,-2,length=10) ) ) 

NNETMODEL <- train(Churn~.,data=TRAIN,method='nnet',tuneGrid=nnetGrid,
                   trControl=fitControl, preProc = c("center", "scale"))
NNETMODEL$results[rownames(NNETMODEL$bestTune),]  #Just the row with the optimal choice of tuning parameter
postResample(predict(NNETMODEL,newdata=HOLDOUT),HOLDOUT$Churn)  #Error on holdout sample for reference
mean(HOLDOUT$Churn=="No")  #Accuracy of naive model that predicts everyone doesn't churn

roc(HOLDOUT$Churn,predict(NNETMODEL,newdata=HOLDOUT,type="prob")[,2])

GBM2.predictions <- predict(GBM2,newdata=HOLDOUT,type="prob")[,2]
sum(GBM2.predictions>=0.5)  #Number of individuals with predicted probabilities greater than 0.5 (a lift of 1.75)
cutoff <- sort(GBM2.predictions,decreasing = TRUE)[2250]  #2250th largest predicted probability;  how many of those are right?
singled.out <- which(GBM2.predictions >= cutoff)
length(singled.out)  #if not 2250, it will pick 2250 at random
singled.out <- sample(singled.out,size=2250) 
table( HOLDOUT$Churn[singled.out] )





#XGBOOST best model
xgboostGrid <- expand.grid(eta=0.001,nrounds=c(250,500),
                           max_depth=9,min_child_weight=1,gamma=1,colsample_bytree=1,subsample=.6)
set.seed(420); XGBOOSTMODEL2 <- train(Churn~.,data=TRAIN,method='xgbTree',tuneGrid=xgboostGrid,
                                      trControl=fitControl, preProc = c("center", "scale"))
XGBOOSTMODEL2$results[rownames(XGBOOSTMODEL2$bestTune),]  #Just the row with the optimal choice of tuning parameter
postResample(predict(XGBOOSTMODEL2,newdata=HOLDOUT),HOLDOUT$Churn)  #Error on holdout sample for reference
mean(HOLDOUT$Churn=="No")  #Accuracy of naive model that predicts everyone doesn't churn

roc(HOLDOUT$Churn,predict(XGBOOSTMODEL2,newdata=HOLDOUT,type="prob")[,2])

XGBOOSTMODEL2.predictions <- predict(XGBOOSTMODEL2,newdata=HOLDOUT,type="prob")[,2]
sum(XGBOOSTMODEL2.predictions>=0.5) 
XGBOOSTMODEL2.predictions[which(XGBOOSTMODEL2.predictions >= .55)]
#Number of individuals with predicted probabilities greater than 0.5 (a lift of 1.75)
cutoff <- sort(XGBOOSTMODEL2.predictions,decreasing = TRUE)[2250]  #2250th largest predicted probability;  how many of those are right?
singled.out <- which(XGBOOSTMODEL2.predictions >= cutoff)
length(singled.out)  #if not 2250, it will pick 2250 at random
singled.out <- sample(singled.out,size=2250) 
table( HOLDOUT$Churn[singled.out] )

XGBOOSTMODEL3$results[rownames(XGBOOSTMODEL3$bestTune),]  #Just the row with the optimal choice of tuning parameter
postResample(predict(XGBOOSTMODEL2,newdata=HOLDOUT),HOLDOUT$Churn)  #Error on holdout sample for reference
mean(HOLDOUT$Churn=="No")  #Accuracy of naive model that predicts everyone doesn't churn

roc(HOLDOUT$Churn,predict(XGBOOSTMODEL2,newdata=HOLDOUT,type="prob")[,2])





treeGrid <- expand.grid(cp=10^seq(-5,-1,length=25))

set.seed(479); TREE <- train(Churn~.,data=TRAIN,method='rpart', trControl=fitControl, tuneGrid=treeGrid,
                               preProc = c("center", "scale"))

TREE$results[rownames(TREE$bestTune),]  #Just the row with the optimal choice of tuning parameter


postResample(predict(TREE,newdata=HOLDOUT),HOLDOUT$Churn)  #Error on holdout sample for reference
mean(HOLDOUT$Churn=="No")  #Accuracy of naive model that predicts everyone doesn't churn

roc(HOLDOUT$Churn,predict(TREE,newdata=HOLDOUT,type="prob")[,2])

#To glimpse how many of the "top" predictions are correct:
TREE.predictions <- predict(TREE,newdata=HOLDOUT,type="prob")[,2]
sum(TREE.predictions>=0.5)  #Number of individuals with predicted probabilities greater than 0.5 (a lift of 1.75)
cutoff <- sort(TREE.predictions,decreasing = TRUE)[2250]  #2250 largest predicted probability;  how many of those are right?
singled.out <- which(TREE.predictions >= cutoff)
length(singled.out)  #if not 2250, it will pick 2250 at random; tree model has too many here :(
singled.out <- sample(singled.out,size=2250) 
table( HOLDOUT$Churn[singled.out] )







#One you have tried out many models, you'll have an idea of which one you want (e.g. a random forest with mtry=10)
#You'll want to fit this on the ENTIRETY of the training data, then predict on the test data

paramGrid2 <- expand.grid(n.trees=c(100,200,500,1000),interaction.depth=4:5,shrinkage=c(.01,.0001),n.minobsinnode=5)  
#update to your model choice and to the exact set of parameters you want
#The final model might take a VERY long time to train
set.seed(479); FINAL <- train(Churn~.,data=TRAIN,method='gbm',trControl=fitControl, tuneGrid = paramGrid2, 
                                      preProc = c("center", "scale"),verbose=FALSE)







#Once you've settled on your final model, make predictions on the test set and write them to a file, to be
#uploaded onto Canvas.
FINAL <- XGBOOSTMODEL2
finalpredictions <- predict(FINAL,newdata=TEST.CLEAN,type="prob")[,2]
length(unique(finalpredictions))  #Should be a lot of unique values!
write(finalpredictions,ncol=1,file="mychurnprobs.csv")





###The following was my GBM work to visualize relationships; not needed here for the prediction contest

library(gbm)  #you'll need to install.packages this first
TRAIN.GBM <- TRAIN.CLEAN  
TRAIN.GBM$Churn <- as.numeric(TRAIN.GBM$Churn) - 1 #Need to set up training set to have 0/1 responses
#This takes LONG
GBM <- gbm(Churn~.,data=TRAIN.GBM,distribution="bernoulli",n.trees=2500,
           interaction.depth=5,shrinkage=0.001,n.minobsinnode = 2)
summary(GBM)  #variable importances
plot(GBM,"OverageMinutes",type="response")  #probability of whatever level is FIRST alphabetically
OUTPUT <- plot(GBM,"OverageMinutes",type="response",return.grid=TRUE)  #probability of whatever level is FIRST alphabetically
HIST <- hist(TRAIN.CLEAN$OverageMinutes,plot=FALSE); 
#Shows how many individuals have different equipment days (histogram) along with probability of churning
plot(HIST$mids,0.2*HIST$counts/max(HIST$counts),type="s",xlab="OverageMinutes",
     ylab="P(churn)",ylim=c(0,0.4))  
lines(y~OverageMinutes,data=OUTPUT,col="red")


plot(GBM,"MonthsInService",type="response")  #probability of whatever level is FIRST alphabetically
OUTPUT <- plot(GBM,"MonthlyMinutes",type="response",return.grid=TRUE)  #probability of whatever level is FIRST alphabetically
hist(TRAIN.CLEAN$MonthlyMinutes,freq=FALSE,ylim=c(0,1),xlab="Monthly Minutes",ylab="P(Churn)"); 
lines(y~MonthlyMinutes,data=OUTPUT)







library(regclass)

library(gbm) #you'll need to install.packages this first
TRAIN.GBM <- EX6.WINE
TRAIN.GBM$Quality <- as.numeric(TRAIN.GBM$Quality == "High") #1=high, 0=low
GBM <- gbm(Quality~.,data=TRAIN.GBM,distribution="bernoulli",n.trees=5000,
           interaction.depth=5,shrinkage=0.001,n.minobsinnode = 2)
summary(GBM2) #variable importances
plot(GBM2,"alcohol",type="response") #probability of whatever level is FIRST alphabetically
OUTPUT <- plot(GBM,"alcohol",type="response",return.grid=TRUE) #probability of whatever level is FIRST alphabetically
HIST <- hist(EX6.WINE$alcohol,plot=FALSE); 
#Shows how many individuals have different equipment days (histogram) along with probability of churning
plot(HIST$mids,0.2*HIST$counts/max(HIST$counts),type="s",xlab="alcohol",
     ylab="P(High)",ylim=c(0,1)) 
lines(y~alcohol,data=OUTPUT,col="red")



