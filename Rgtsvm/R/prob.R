# source code
# http://www.csie.ntu.edu.tw/~cjlin/papers/plattprob.pdf
#
# Input parameters:
# deci = array of SVM decision values
# label = array of booleans: is the example labeled +1?
# prior1 = number of positive examples
# prior0 = number of negative examples
# Outputs:
# A, B = parameters of sigmoid

optim_Platt <- function(deci, label, prior1, prior0)
{
	## Maximum number of iterations
	maxiter=100
	## Minimum step taken in line search
	minstep=1e-10
	## Set to any value > 0
	sigma=1e-12

	## Construct initial values: target support in array t,
	## initial function value in fval
	hiTarget <- (prior1+1.0)/(prior1+2.0);
	loTarget=1/(prior0+2.0);

	## Total number of data
	len <- prior1+prior0
	t <- rep(NA, len);
	for (i in 1:len) {
		if (label[i] > 0)
			t[i] <- hiTarget
		else
			t[i] <- loTarget
	}

	A <- 0.0;
	B <- log((prior0+1.0)/(prior1+1.0));
	fval <- 0.0

	for (i in 1:len)
	{
		fApB <- deci[i]*A+B;
		if (fApB >= 0)
			fval <- fval + t[i]*fApB+log(1+exp(-fApB))
		else
			fval <- fval + (t[i]-1)*fApB+log(1+exp(fApB))
	}

	bFound <- FALSE;
	for (it in 1:maxiter) {
		## Update Gradient and Hessian (use H’ = H + sigma I)
		h11 <- h22 <- sigma;
		h21 <- g1 <- g2 <- 0.0;
		for (i in 1:len)
		{
			fApB <- deci[i]*A+B
			if (fApB >= 0)
			{
				p <- exp(-fApB)/(1.0+exp(-fApB));
				q <- 1.0/(1.0+exp(-fApB));
			}
			else
			{
				p <- 1.0/(1.0+exp(fApB));
				q <- exp(fApB)/(1.0+exp(fApB));
			}

			d2  <- p*q;
			h11 <- h11 + deci[i]*deci[i]*d2;
			h22 <- h22 + d2;
			h21 <- h21 + deci[i]*d2;
			d1 <- t[i]-p;
			g1 <- g1 +  deci[i]*d1;
			g2 <- g2 + d1;
		}

		##Stopping criteria
		if (abs(g1)<1e-5 && abs(g2)<1e-5)
		{
			bFound <- TRUE;
			break
		}

		## Compute modified Newton directions
		det <- h11*h22-h21*h21;
		dA  <- -(h22*g1-h21*g2)/det;
		dB  <- -(-h21*g1+h11*g2)/det;
		gd  <- g1*dA+g2*dB;
		stepsize <- 1;

		while (stepsize >= minstep)
		{   ## Line search
			newA <- A+stepsize*dA;
			newB <- B+stepsize*dB;
			newf <- 0.0;
			for (i in 1:len) {
				fApB <- deci[i]*newA+newB
				if (fApB >= 0)
					newf <- newf + t[i]*fApB+log(1+exp(-fApB))
				else
					newf <- newf + (t[i]-1)*fApB+log(1+exp(fApB))
			}

			if (newf<fval+0.0001*stepsize*gd)
			{
				A <- newA; B <- newB; fval<- newf;
				bFound <- TRUE;
				## Sufficient decrease satisfied
				break;
			}
			else
				stepsize <- stepsize / 2.0
		}

		if (stepsize < minstep)
		{
			cat("Line search fails\n")
			break;
		}
	}

	if (it >= maxiter)
		cat("Reaching maximum iterations");

	if(bFound)
		return(list(A=A,B=B))
	else
		return(NULL);
}

svc_binary_train_prob <- function( ylabel, ydeci )
{
	prior1 <- sum( ylabel == 1 );
	prior0 <- sum( ylabel != 1 );
	ret <- optim_Platt( ydeci, ylabel, prior1, prior0)
	return(ret);
}

svc_binary_predict_prob <- function( ydeci,A, B )
{
	ret <- rep(0,length(ydeci))
	fApB <- ydeci*A +B
	bindex <- fApB >= 0
	ret[ bindex ] <- exp(-fApB[bindex])/(1 + exp(-fApB[bindex]))
	ret[ !bindex ] <- 1/(1 + exp(fApB[!bindex]))

	return(ret);
}

svc_one_again_all_train_prob <- function( ylabel, ydeci )
{
	fn <- function(r, ylabel, ydeci)
	{
		if(max(ylabel)<NCOL(ydeci) )
			y <- ydeci[1:NROW(ydeci) + ylabel*NROW(ydeci) ]
		else
			y <-  ydeci[1:NROW(ydeci) + (ylabel-1)*NROW(ydeci) ];

		R <- r*y - log(rowSums(exp(r*ydeci)));
		return( -sum(R) );
	}

	r.init <- 1
	r <- optim(r.init, fn, ylabel=ylabel, ydeci=ydeci, method="Brent", lower=0, upper=100);

	if (r$convergence==0)
		return(r$par)
	else
		return(NA);
}

svc_one_again_all_predict_prob <- function( ydeci, r )
{
	p0 <- exp(r*ydeci);
	P <- p0/matrix( rowSums(p0), nrow=NROW(ydeci), ncol=NCOL(ydeci));

	return(P);
}

## Return parameter of a Laplace distribution
eps_train_prob <- function( y, ypred, cross=0 )
{
	std <- sqrt( 2 * sum( abs(ypred - y)/NROW(y) ) ^2 );
	outlier <-  which(abs(ypred - y) > 5*std);
	if(length(outlier)>0)
		mae <- mean( abs(ypred - y)[ - outlier] )
	else
		mae <- mean( abs(ypred - y) )

	cat ("Prob. model for test data: target value = predicted value + z,\nz: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma=", mae, "\n" );
	return(mae);
}

eps_predict_prob <- function( ypred, mae )
{
}

