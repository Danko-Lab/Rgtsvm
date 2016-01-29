library(Rgtsvm);

x1 <- seq(0.1, 5, by = 0.01)
x2 <- rnorm(x1, sd = 0.2)
y <- log(x1) + rnorm(x2 +x1, sd = 0.2)

gt.eps1   <- svm( cbind(x1,x2), y, gamma=0.5, type="eps-regression");

y.pre <- predict( gt.eps1, cbind(x1,x2));

MSE <- sum((y.pre-y)^2)/length(y);
cat("MSE=", MSE, "\n");

gt.eps2   <- svm( cbind(x1,x2), y, gamma=0.5, type="eps-regression", cross=10);

data <-data.frame(Y=y, cbind(x1,x2) );
gt.eps3   <- svm( Y~., data, gamma=0.5, type="eps-regression");

plot(gt.eps3, data);


load("training_x_y.RData");
#gt.train   <- svm( dnase.vector~., training_x_y, gamma=0.00833, type="eps-regression");

gt.train   <- svm( training_x_y[,-c(121)], training_x_y[,121]*10, gamma=0.00833, type="eps-regression", scale=TRUE);
