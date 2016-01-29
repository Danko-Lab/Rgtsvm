library(Rgtsvm);

## more than two variables: fix 2 dimensions
data(iris)

m2 <- svm(Species~., data = iris)
p2 <- predict(m2, iris[, - 5] );

plot(m2, iris, Petal.Width ~ Petal.Length, slice = list(Sepal.Width = 3, Sepal.Length = 4))
