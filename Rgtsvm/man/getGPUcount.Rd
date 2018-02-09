\name{getGPUcount}
\alias{getGPUcount}
\title{
Checking the count of GPU device.
}
\usage{
	getGPUcount()
}
\value{
The count of GPU device.
}
\seealso{
\code{\link{selectGPUdevice}},\code{\link{resetGPUdevice}},
}
\examples{
cat("GPU count:", getGPUcount(), "\n");
}
\keyword{ Multiple GPU }
