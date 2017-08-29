\name{load.bigmatrix}
\alias{load.bigmatrix}
\title{
Loading a big matrix from RData or RDS file
}
\description{
Creating a big matrix based on the matrix variable in a RData file or a RDS file.
}
\usage{
load.bigmatrix(file.data, variable = NULL)
}
\arguments{
  \item{file.data}{File name, RData file or RDS file.}
  \item{variable}{String, variable name in the RData file. If variable is NULL, the data file should be in RDS format.}
}
\value{
Return a reference class with the name "BigMatrix.refer". It can be used in \code{\link{svm}} and \code{\link{predict}} calling in Rgtsvm.
}
\seealso{
\code{\link{attach.bigmatrix}}
}
\examples{
# The example can not be executed!
#   
#  x0_bm <- load.bigmatrix("X0.RDS")
#  x1_bm <- load.bigmatrix("X1.Rdata", "x1")

}
\keyword{ data loading }

