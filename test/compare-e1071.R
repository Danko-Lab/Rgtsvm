library(e1071);

#load("MAZ_svmdata.rdata");   # TIME: 4:40:00
load("ELF1_trainData_defAdd100010.rdata");

trainAllStatus[trainAllStatus==0] <- -1;

#if(file.exists("test1-e.rdata"))
#{
#	load("test1-e.rdata");
#}else{
	ptm <- proc.time()
	model.e <- e1071::svm( x=trainAll, y=trainAllStatus, type="C-classification", fitted=FALSE);
	pt <- proc.time() - ptm
	show(pt);

	save(pt, model.e, file="test1-e.rdata");
#}

predict.e <- e1071::predict( model.e, trainAll);

save( predict.e, file="test1-e-pre.rdata");

z <- ( as.character(predict.e) ==as.character(trainAllStatus))

correct <- length(which(z))/length(z);

cat("correct=", correct, "\n");
