library(Rgtsvm);
library(SparseM)

load("ELF1_trainData.rdata"); # 99.5% (*)23 sec
#load("MAZ_svmdata.rdata")    # 99.4%
#load("389k_svmdata.rdata")   # 96.0% (*)20 min
#load("680k_svrdata.rdata")   # 97.1% (*)63 min 
#A<-as.matrix.csr(trainAll);

gt.model <- svm( x=trainAll, y=trainAllStatus, gamma=0.05, type="C-classification", scale = FALSE, fitted=TRUE);
save(trainAllStatus, gt.model, file="test1-gt.rdata");

cat("correct=", gt.model$correct, "\n");

gt.predit <- predict( gt.model, trainAll );
save( trainAllStatus, gt.model, gt.predit, file="test1-gt.rdata");