\name{selectGPUdevice}
\alias{selectGPUdevice}
\title{
Selecting GPU device to run training or prediction.
}
\description{
It is useful to select one GPU devices to run if multiple GPU devices are on the host.  
}
\usage{
selectGPUdevice( gpu.id )
}
\arguments{
  \item{gpu.id}{integer value indicating GPU device ID, starting from 0. No GPU device is changed by default. }
}
\value{
This function returns error code, which 0 indicates success otherwise indicates failure.
}
\seealso{
\code{\link{svm}},\code{\link{predict.gtsvm}}, \code{\link{predict.loadsvm}}
}
\examples{
	selectGPUdevice(0);
}
\keyword{ Multiple GPU }
