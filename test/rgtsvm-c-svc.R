library(Rgtsvm);

load("ELF1_trainData.rdata"); # 99.5% (*)23 sec
#load("MAZ_svmdata.rdata")    # 99.4%
#load("389k_svmdata.rdata")   # 96.0% (*)20 min
#load("680k_svrdata.rdata")   # 97.1% (*)63 min 

trainAllStatus[trainAllStatus==0] <- -1;

idx1 <- which(trainAllStatus==1)
idx2 <- which(trainAllStatus==-1)
trainAll <- trainAll[c(idx1, idx2),]
trainAllStatus <- trainAllStatus[c(idx1, idx2)]

data <- data.frame(Y=trainAllStatus, trainAll);

gt.model1 <- svm( Y~., data, gamma=0.05, tolerance=0.01, type="C-classification", scale = FALSE, fitted=TRUE);

gt.model2 <- svm( Y~., data, gamma=0.05, tolerance=0.01, type="C-classification", scale = TRUE, fitted=TRUE, cross=10);

gt.predict2 <- predict( gt.model2, trainAll );
accuracy <- length(which(trainAllStatus==gt.predict2))/length(trainAllStatus);
cat("accuracy=", accuracy, "\n");

save( gt.model1, gt.model2, file="test1-gt1.rdata");

plot(gt.model2, data, formula = Y~.);

