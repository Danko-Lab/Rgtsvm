library(dREG);

load("test1-gt2.rdata")

roc_mat <- logreg.roc.calc( statusBU, gt.predit );
AUC<- roc.auc(roc_mat);
roc.plot(roc_mat, main=AUC );
