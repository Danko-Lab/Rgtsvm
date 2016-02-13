library(Rgtsvm);

load("ELF1_trainData.rdata"); # 99.5% (*)23 sec
data <- data.frame(Y=trainAllStatus, trainAll);

### step1: tuning paramaters
gt.tune <- tune.svm( trainAll, trainAllStatus, gamma = 2^c(-8, -6 ), cost = 2^c(-1, 0, 1, 2), tunecontrol=tune.control(sampling = "cross", cross = 10,rough.cross=2) );
#check the plot to see the best area
plot(gt.tune, transform.x = log2, transform.y = log2);
#check the best parameters 
show(gt.tune);
show(gt.tune$performances);

save( gt.tune, file="gt.tune.rdata");

### Step 2:  run model with cross-validation using the best parameteers tuned by tune.svm
gt.model <- svm(trainAll, trainAllStatus, type="C-classification", gamma=0.00390625, cost=4, fitted=FALSE, cross=10, rough.cross=2 );

## check the accuracy calculacted by cross-validation
cat(gt.model$tot.accuracy );
cat(gt.model$accuracies );

### Step 3: predict using the training model
ret <- predict(gt.model, trainAll, decision.values=TRUE);

