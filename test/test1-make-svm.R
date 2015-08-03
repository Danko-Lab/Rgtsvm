load("ELF1_trainData.rdata");

trainAllStatus[trainAllStatus==0] <- -1;

x<-array( which(!is.na(trainAll), arr.ind = T)[,2], dim=dim(trainAll))
s <- array( paste(x, trainAll, sep=":"), dim=dim(trainAll))
df <- cbind(trainAllStatus, s);

write.table(df, file="test1-gt.dat", quote=F, row.names=F, col.names=F, sep=" ");
