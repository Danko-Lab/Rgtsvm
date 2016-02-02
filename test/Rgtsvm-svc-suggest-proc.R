library(Rgtsvm);

load("ELF1_trainData.rdata"); # 99.5% (*)23 sec
data <- data.frame(Y=trainAllStatus, trainAll);

#step1: tuning paramaters
gt.tune <- tune.svm( Y~., data=data, sampling = "cross", gamma = 2^c(-8, -6, -4, -2), cost = 2^c(-1, 0, 1, 2, 3), cross=10 );
save( gt.tune, file="gt.tune.rdata");

#check the plot to see the best area
plot( gt.tune, transform.x = log2, transform.y = log2)

#check the best parameters 
show(gt.tune);

#run model with crossvalidation using the best parameteers tuned by tune.svm
gt.model <- svm( Y~., data=data, type="C-classification", cross=10, gamma=0.00390625, cost=8, fitted=FALSE );

## check the accuracy calculacted by cross-validation
cat(gt.model$tot.accuracy );
cat(gt.model$accuracies );

## predict using the training model
ret <- predict(gt.model, trainAll, decision.values=TRUE);