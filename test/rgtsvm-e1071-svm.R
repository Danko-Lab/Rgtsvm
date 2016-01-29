library(Rgtsvm)
data(iris)
attach(iris)
     
## classification mode
# default with factor response:
model <- svm(Species ~ ., data = iris)
     
# alternatively the traditional interface:
x <- subset(iris, select = -Species)
y <- Species
model <- svm(x, y) 
     
print(model)
summary(model)
     
# test with train data
pred <- predict(model, x)
# (same as:)
pred <- fitted(model)
     
# Check accuracy:
table(pred, y)
     
# compute decision values and probabilities:
pred <- predict(model, x, decision.values = TRUE)
attr(pred, "decision.values")[1:4,]
     
# visualize (classes by color, SV by crosses):
plot(cmdscale(dist(iris[,-5])),
     col = as.integer(iris[,5]),
     pch = c("o","+")[1:150 %in% model$index + 1])
     


## tune `svm' for classification with RBF-kernel (default in svm),
## using one split for training/validation set

obj <- tune(svm, Species~., data = iris, ranges = list(gamma = 2^(-1:1), cost = 2^(2:4)), 
	tunecontrol = tune.control(sampling = "fix");
)

obj <- tune(svm, Species~., data = iris, ranges = list(gamma = 2^(-1:1), cost = 2^(2:4)), 
	tunecontrol = tune.control(sampling = "cross", cross=5);
)

## alternatively:
## obj <- tune.svm(Species~., data = iris, gamma = 2^(-1:1), cost = 2^(2:4))     

