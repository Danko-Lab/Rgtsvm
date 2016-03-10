file.rdata <- "ELF1_trainData.rdata";
#file.rdata <- "/work/03350/tg826494/Rgtsvm/test/chipseq_5010_k562_dnase_training.rdata";

library(Rgtsvm);

### Example 1

load(file.rdata); 
bigm.x <- attach.bigmatrix(data = trainAll)
gt.model <- svm( bigm.x, trainAllStatus, gamma=0.00005, tolerance=0.001, type="C-classification", scale = TRUE, fitted=TRUE, cross=4);

#NOTICE: please load the training data again because it has been changed by reference calling.
load(file.rdata); 
bigm.x <- attach.bigmatrix(data = trainAll)
gt.predict <- predict( gt.model, bigm.x);

accuracy <- length(which(trainAllStatus==gt.predict))/length(trainAllStatus);
cat("accuracy=", accuracy, "\n");

plot(gt.model, data, formula = Y~.);


### Example 2

load(file.rdata);
trainAll <- NULL;
gc();

bigm.x <- load.bigmatrix(file.rdata, variable="trainAll");

gt.tune <- tune.svm( bigm.x, trainAllStatus, gamma = 2^c(-6, -8, -10), cost = 2^c(0.5, 1.5,  2.5), verbose=F, tolerance = 1,
	tunecontrol=tune.control(sampling = "cross", cross = 10, rough.cross=1 ) );

pdf("rgtsvm-bigm.pdf");
plot( gt.tune, transform.x = log2, transform.y = log2);
dev.off();

show(gt.tune);

