library(Rgtsvm);
library(SparseM)

load("ELF1_trainData.rdata"); # 99.5% (*)23 sec
#load("/work/03350/tg826494/Rgtsvm/test/MAZ_svmdata.rdata")    # 99.4%
#load("/work/03350/tg826494/Rgtsvm/test/389k_svmdata.rdata")   # 96.0% (*)20 min
#load("/work/03350/tg826494/Rgtsvm/test/680k_svrdata.rdata")   # 97.1% (*)63 min 
#load("/work/03350/tg826494/Rgtsvm/test/400k_trainingvectors.rdata") 
#load("/work/03350/tg826494/Rgtsvm/test/MAFK_chipseqTesting_k562_gray.rdata") 

trainAll <- testAll;
trainAllStatus <- testAllStatus;

gt.model <- svm( x=trainAll, y=trainAllStatus, gamma=0.05, type="C-classification", scale = FALSE, fitted=TRUE, tolerance=0.01);
gt.model <- svm( x=trainAll, y=trainAllStatus, gamma=0.05, type="C-classification", scale = TRUE, fitted=TRUE, tolerance=0.01, cross=5);

#save(trainAllStatus, gt.model, file="test1-gt.rdata");

cat("correct=", gt.model$fitted.accuracy, "\n");

gt.predit <- predict( gt.model, trainAll );

accuracy <- length(which(gt.predit==trainAllStatus))/length(trainAllStatus)
cat("predict=", accuracy, "\n");

#save( gt.model, gt.predit, trainAllStatus, file="test1-gt.rdata");
