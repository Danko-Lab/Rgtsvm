load("ELF1_trainData.rdata");

trainAllStatus[trainAllStatus==0] <- -1;

x <- array( which(!is.na(trainAll), arr.ind = T)[,2], dim=dim(trainAll))
s <- array( paste(x-1, trainAll, sep=":"), dim=dim(trainAll))
df <- data.frame(trainAllStatus, s);

write.table(df, file="test1-gt.dat", quote=F, row.names=F, col.names=F, sep=" ");


library(Rgtsvm);

mat <- load.svmlight("test1-gt.dat")
 
gt.model <- svm( x=mat[,-1], y=mat[,1], gamma=0.05, tolerance=0.01, type="C-classification", scale = FALSE, fitted=TRUE, cross=10);
save(gt.model, file="test2-gt.rdata");
cat("correction=", gt.model$correct, "\n");

gt.predit <- predict( gt.model, mat[,-1], score=FALSE );
save( gt.model, gt.predit, y=mat[,1], file="test2-gt.rdata");

correct = length( which(gt.predit == mat[,1]) )/nrow(mat)
cat("correct=", correct, "\n");

summary(gt.predit);


gt.model2 <- svm( x=mat[,-1], y=mat[,1], gamma=1/ncol(mat[,-1]), tolerance=0.001, type="C-classification", scale = TRUE, fitted=F);