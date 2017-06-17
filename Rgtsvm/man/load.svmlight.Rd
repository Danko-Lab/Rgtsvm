\name{load.svmlight}
\alias{load.svmlight}
\title{
Load SVMlight data file into a sparse matrix.
}
\description{
Load SVMlight data file into a sparse matrix.
}
\usage{
load.svmlight(filename, .loadbyC = TRUE)
}
\arguments{
  \item{filename}{string, SVM light filename.}
  \item{.loadbyC}{logical value, indicating whether loading data in C or R.}
}
\details{
The file must be svmlight format.(\href{http://svmlight.joachims.org/}{http://svmlight.joachims.org/})
}
\value{
A sparse matrix is returned if the file is loaded or downloaded successfully.
}
\author{
  Zhong Wang ( R interface )  \email{zw355@cornell.edu}\cr
}
\examples{
mat <-load.svmlight("http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/glass.scale")
str(mat);
}
\keyword{ data loading }
