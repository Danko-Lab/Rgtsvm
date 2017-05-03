# qsub -I -l nodes=1:ppn=20 -q hybrid -d .

library(Rgtsvm);

# Downloading epsilon data from
# https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/epsilon_normalized.t.bz2
# and
# https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/epsilon_normalized.bz2
# and then uncompress data using the command 'bunzip2'

# Loading data in the SVM light format.
# Notice: the orignial training data is too large as a demonstration,
#   therefore we use the test dataset for this demo.
dat <- load.svmlight("epsilon_normalized.t")

# Shuffling the indexes and using 60K as training dataset and
#   the reset as test dataset
idx <- sample(1:NROW(dat))
i.train <- idx[1:60000];
i.test <- idx[-c(1:60000)];

# tuning parameters including gamma and cost
# Notice: the cross and cross number and other parameters should be assigned
#    into 'tune.control' structure.
gt.tune <- tune.svm( dat[i.test,-c(1,2)], dat[i.test,1],
  gamma = 2^seq(-11, -1, 2),
  cost = 10^(-1:1),
  tunecontrol=tune.control( sampling = "cross", cross=8, rough.cross=3), scale=F );
save(gt.tune, i.train, i.test, file="gt.tune.rdata");

# Printing the tuning results a). best model and b). best parameters
show(gt.tune$best.model);
show(gt.tune$best.parameters);

# Drawing the figure of tuning results.
pdf("svm-tune.pdf");
plot( gt.tune, transform.x = log2, transform.y = log2)
plot( gt.tune, type = "perspective", theta = 120, phi = 45)
dev.off();

#gt.svm <- svm( dat[i.train,-c(1,2)], dat[i.train,1], gamma=1/8, cost=1, cross=8);
# Training the model using the best parameters.
gt.svm <- svm( dat[i.train,-c(1,2)], dat[i.train,1],
  gamma=as.numeric(gt.tune$best.parameters[1]),
  cost=as.numeric(gt.tune$best.parameters[2]),
  scale=F);

# Showing the trained model and its accuracy
show(gt.svm);
cat("Accuracy for the training data=", gt.svm$fitted.accuracy, "\n");

# Predicting the test dataset, using 'decision.values' to
#   get the decision values
y.pred <- predict( gt.svm, dat[i.test,-c(1,2)], decision.values=TRUE )

# Showing the accuracy for the test dataset
table(y.pred==dat[i.test,1]);
cat("Accuracy for the test data=",
   length(which( y.pred == dat[i.test,1] ) )/length(y.pred), "\n" );

# Download the script to draw PR or ROC curve to verify the model quality
source("https://raw.githubusercontent.com/andybega/auc-pr/master/auc-pr.r");

# Using the decision values to draw the PR and ROC curve.
str(y.pred);
pred<-attr(y.pred, "decision.values");

pdf("svm-eval.pdf");
# Drawing PR curve and calculate the AUC value
xy <- rocdf(pred, obs=dat[i.test,1], type="pr")
AUC <- auc_pr( obs=dat[i.test,1], pred)
plot(xy[, 1], xy[, 2], xlab="Recall", ylab="Precision",
   cex.lab=1.5,cex.axis=1.5,
   main=paste("PR AUC=", round(AUC,3)));

# Drawing ROC curve and calculate the AUC value
xy <- rocdf(pred, obs=dat[i.test,1], type="roc")
AUC <- auc_roc( obs=dat[i.test,1], pred)
plot(xy[, 1], xy[, 2], xlab="False Positive Rates",
   ylab="True Positive rates", cex.lab=1.5,cex.axis=1.5,
   main=paste("ROC AUC=", round(AUC,3)));

dev.off();

