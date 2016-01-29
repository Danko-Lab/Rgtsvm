library(Rgtsvm);

data(cats, package = "MASS")
gamma <- 1 / (ncol(cats)-1)
m <- svm(Sex~., data = cats, gamma=gamma)
plot(m, cats)

## more than two variables: fix 2 dimensions
data(iris)

gamma <- 1 / (ncol(iris)-1)
m2 <- svm(Species~., data = iris, gamma=gamma)

plot(m2, iris, Petal.Width ~ Petal.Length, slice = list(Sepal.Width = 3, Sepal.Length = 4))

## plot with custom symbols and colors
plot(m, cats, svSymbol = 1, dataSymbol = 2, symbolPalette = rainbow(4), color.palette = terrain.colors)

dev.off();
