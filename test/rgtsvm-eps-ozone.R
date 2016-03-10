library(Rgtsvm)
data(Ozone, package="mlbench");

index <- 1:NROW(Ozone);
testindex <- sample(index, trunc(length(index)/3))
testset <- na.omit(Ozone[testindex, -3]);
trainset <- na.omit(Ozone[-testindex, -3]);

svm.model <- svm(V4~., data=trainset, cost=1000, gamma=1e-3, type="eps-regression")
svm.pred <- predict(svm.model, testset[,-3]);
crossprod(svm.pred - testset[,3])/length(testindex);
