library(Rgtsvm);

file.rdata <- "ELF1_trainData.rdata"; # 99.5% (*)23 sec

load( file.rdata ); 
trainAllStatus[trainAllStatus==0] <- -1;
idx1 <- which(trainAllStatus==1)
idx2 <- which(trainAllStatus==-1)
trainAll <- trainAll[c(idx1, idx2),]
trainAllStatus <- trainAllStatus[c(idx1, idx2)]
data <- data.frame(Y=trainAllStatus, trainAll);

### Example 1
gt.model1 <- svm( Y~., data=data, gamma=0.05, tolerance=0.01, type="C-classification",class.weights=c(0.5, 200), scale = FALSE, fitted=TRUE);

gt.predict1 <- predict( gt.model1, trainAll, decision.values = TRUE );
accuracy <- length(which(trainAllStatus==gt.predict1))/length(trainAllStatus);
cat("accuracy=", accuracy, "\n");

plot(gt.model1, data, formula = Y~.);

### Example 2

load( file.rdata ); 

gt.model2 <- svm( trainAll, trainAllStatus, gamma=0.00390625, cost=2, type="C-classification", scale = TRUE, fitted=TRUE, cross=20, rough.cross=4);

gt.predict2 <- predict( gt.model2, trainAll, decision.values = TRUE );
accuracy <- length(which(trainAllStatus==gt.predict2))/length(trainAllStatus);
cat("accuracy=", accuracy, "\n");


### Example 3
train.id <- sample(1:NROW(trainAll))[1:round(NROW(trainAll)*0.85)]
gt.model1 <- svm( trainAll[train.id,], trainAllStatus[train.id], gamma=0.00390625, cost=2, type="C-classification", scale = TRUE, fitted=TRUE);
gt.predict1 <- predict( gt.model1, trainAll[-train.id,], decision.values = TRUE );
accuracy <- length(which(trainAllStatus[-train.id]==gt.predict1))/length(trainAllStatus[-train.id]);
cat("accuracy=", accuracy, "\n");
