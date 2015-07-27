library(Rgtsvm);
library(SparseM)

load("ELF1_trainData.rdata"); #99.5%
#load("MAZ_svmdata.rdata")  # 99.4%
#load("389k_svmdata.rdata") # 96.01% 20 min
#load("680k_svrdata.rdata") # 97.1% 62.5 min 

trainAllStatus[trainAllStatus==0] <- -1;

idx1 <- which(trainAllStatus==1)
idx2 <- which(trainAllStatus==-1)
trainAll <- trainAll[c(idx1, idx2),]
trainAllStatus <- trainAllStatus[c(idx1, idx2)]

A<-as.matrix.csr(trainAll);

ptm <- proc.time()
gt.model <- svm( x = A, y = trainAllStatus, gamma=0.05, tolerance=0.01, type="C-classification", scale = FALSE, fitted=FALSE);
pt1 <- proc.time() - ptm
show(pt1);

save(pt1, gt.model, file="test1-gt1.rdata");

sampling.idx <- sample(1:length(trainAllStatus));
trainAll <- trainAll[ sampling.idx,]
trainAllStatus <- trainAllStatus[ sampling.idx ]

B<-as.matrix.csr(trainAll);

ptm <- proc.time()
gt.predit <- predict( gt.model, B );
pt2 <- proc.time() - ptm
show(pt2);

correct <- length(which( ( gt.predit * trainAllStatus )==1))/length(trainAllStatus);
cat("correct=", correct, "\n");

save(pt2, gt.predit, file="test1-gt2.rdata");
