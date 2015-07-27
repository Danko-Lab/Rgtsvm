library(Rgtsvm);

load("ELF1_trainData.rdata"); # 99.5% (*)23 sec
#load("MAZ_svmdata.rdata")    # 99.4%
#load("389k_svmdata.rdata")   # 96.0% (*)20 min
#load("680k_svrdata.rdata")   # 97.1% (*)63 min 

#A<-as.matrix.csr(trainAll);

trainAllStatus[trainAllStatus==0] <- -1;

idx1 <- which(trainAllStatus==1)
idx2 <- which(trainAllStatus==-1)
trainAll <- trainAll[c(idx1, idx2),]
trainAllStatus <- trainAllStatus[c(idx1, idx2)]

data <- data.frame(Y=trainAllStatus, trainAll);

ptm <- proc.time()
gt.model <- svm( Y~., data, gamma=0.05, tolerance=0.01, type="C-classification", scale = FALSE, fitted=TRUE);
pt1 <- proc.time() - ptm
show(pt1);

save(pt1, data, gt.model, file="test1-gt1.rdata");

plot(gt.model, data, formula = Y~.);

