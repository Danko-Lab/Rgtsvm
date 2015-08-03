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
     


     

