CACHE_SIZE <-  1000*1000*20;

#BigMatrix.refer <- setRefClass("BigMatrix.refer",

setRefClass("BigMatrix.refer",

	fields = list( data  = "matrix", 
			file.rdata   = "character", 
			variable     = "character",
			scale        = "list",
			nac          = "list",
			stacks       = "list",
			file.backup  = "character",
			row.index    = "numeric",  ## SAVE in the stacks when push or pop
			col.index    = "numeric",  ## SAVE in the stacks when push or pop
			row.subset   = "numeric",  ## SAVE in the stacks when push or pop
			col.subset   = "numeric"), ## SAVE in the stacks when push or pop

    methods = list(
		#edit = function(i, j, value) 
		#{
        # 	data[i,j] <<- value;
        # 	invisible(value)
    	#},
     	
		show = function() 
		{
       		cat("Reference matrix object of class",	classLabel(class(.self)), "\n")
	        cat("Original Data: [", base::NROW(data), base::NCOL(data), "]\n");
	        cat("Reshaped Data: [", row.subset, col.subset, "]\n");

	        x.col <- col.subset;
	        if (x.col>5) x.col<-5;
	        x.row <- row.subset;
	        if (x.row>5) x.row<-5;
	        
	        methods::show( data[ row.index[1:x.row], col.index[1:x.col] ]);
	    },
	     
		NROW = function()
		{
			return(row.subset);
		},
		
		NCOL = function()
		{
			return(col.subset);
		},

		nrow = function()
		{
			return(row.subset);
		},

		ncol = function()
		{
			return(col.subset);
		}
     )
)

setGeneric("bigm.internal.nrow",
	def = function(x){
		stopifnot(class(x) == "BigMatrix.refer");
		standardGeneric("bigm.internal.nrow");
	});

setMethod("bigm.internal.nrow", "BigMatrix.refer", function(x){
	return(NROW(x$data));	
	});

setGeneric("bigm.internal.ncol",
	def = function(x){
		stopifnot(class(x) == "BigMatrix.refer");
		standardGeneric("bigm.internal.ncol");
	});

setMethod("bigm.internal.ncol", "BigMatrix.refer", function(x){
	return(NCOL(x$data));	
	});

setMethod("as.double", "BigMatrix.refer", function(x){ as.double(x$data); } )

setMethod("is.matrix", "BigMatrix.refer", function(x){ return(TRUE); } )

setMethod("[",    "BigMatrix.refer", function(x, i, j,..., drop){ 
	if (missing(i))
		x$data[ , x$col.index[j], drop=drop]
	else
	if (missing(j))
		x$data[ x$row.index[i], , drop=drop]
	else		
		x$data[ x$row.index[i], x$col.index[j], drop=drop];
	} );

setMethod("[<-",  "BigMatrix.refer", function(x, i, j,value){

	x$data[ x$row.index[i], x$col.index[j] ] <- value;
	x$data;
	invisible();

	} );

setMethod("dim", "BigMatrix.refer", function(x){return(c(x$row.subset, x$col.subset));} )

#setGeneric("rowSums",
#	def = function(x, na.rm = FALSE, dims = 1){
#		stopifnot(class(x) == "BigMatrix.refer");
#		standardGeneric("rowSums");
#	})

setMethod("rowSums", "BigMatrix.refer", function(x, na.rm = FALSE, dims = 1)
{
	return(rowSums(x$data[x$row.index[1:x$row.subset], ], na.rm=na.rm, dims=dims));
});


#setGeneric("colSums",
#	def = function(x, na.rm = FALSE, dims = 1){
#		stopifnot(class(x) == "BigMatrix.refer");
#		standardGeneric("colSums");
#	})

setMethod("colSums", "BigMatrix.refer", function(x, na.rm = FALSE, dims = 1)
{
	return(colSums(x$data[,x$col.index[1:x$col.subset]], na.rm=na.rm, dims=dims));
});


setGeneric("bigm.scale",
	def = function(x, scaled.cols, center=NULL, scale=NULL){
		stopifnot(class(x) == "BigMatrix.refer");
		standardGeneric("bigm.scale");
	});

setMethod("bigm.scale", "BigMatrix.refer", function(x, scaled.cols, center=NULL, scale=NULL){

		if ( length(which(scaled.cols))*nrow(x$data) < CACHE_SIZE )
		{
			if(is.null(center) && is.null(scale) )
			{
				xtmp <- scale( x$data[,x$col.index[scaled.cols] ])
				x$data[,x$col.index[scaled.cols] ] <- xtmp
				x.scale <- attributes(xtmp)[c("scaled:center","scaled:scale")]			
				return(x.scale);
			}
			else
			{
	         	x$data[, scaled.cols] <-
	         	   scale( x$data[, scaled.cols, drop = FALSE], center = center, scale  = scale );

				invisible(return(x));
			}
		}	


		col.n <- round( CACHE_SIZE / nrow(x$data) );
		col.pool <- c();

		if(is.null(center) && is.null(scale) )
		{
			center <-c();
			scale <-c();

			for(i in 1:length(scaled.cols))
			{
				if (scaled.cols[i])
					col.pool <- c(col.pool, i);

				if( length(col.pool) >= col.n || i==length(scaled.cols))
				{
					xtmp <- scale( x$data[, x$col.index[col.pool], drop=F] );

					## NOTICE:
					## 
					## althogh r.dummy include 3 objects, but they are not duplicated.
					## Check by th follow methods:
					## lsos(envir=environment());
					## gc();
					## .Internal(inspect( r.dummy[[1]]) )
					## 
					## address = function(x) substring(capture.output(.Internal(inspect(x)))[1],2,17)
					## address(x$data)==address( r.dummy[[1]]) 
					## [1] TRUE

					#### x$data[,x$col.index[col.pool]] <- xtmp
					vec.cols <- x$col.index[col.pool];
					r.dummy  <- .C("bigmatrix_set_bycols", x$data, as.integer(NROW(x$data)), as.integer(NCOL(x$data)), as.integer(vec.cols), as.integer(NROW(vec.cols)), as.double(xtmp), NAOK = TRUE, DUP = FALSE, PACKAGE="Rgtsvm");

					center <- c(center,  unlist( attributes(xtmp)[c("scaled:center" )] ));
					scale  <- c(scale,   unlist( attributes(xtmp)[c("scaled:scale" )] ));
					col.pool <- c();
				}
			}

			names(scale)<-NULL;
			names(center)<-NULL;
			x$scale$scaled.cols <- scaled.cols;
			x$scale$scaled.center <- center;
			x$scale$scaled.scale <- scale;
			
			return( list(`scaled:center`=center, `scaled:scale`=scale) );
		}
		else
		{
			if( length( which(scaled.cols)) != length(center) || length(scale) != length(center) )
				stop("scale columns dont have same length center and scale values.");
			
			scale.pool <- c();
			scale.cursor <- 1;
		
			for(i in 1:length(scaled.cols))
			{
				if (scaled.cols[i])
				{
					col.pool <- c(col.pool, i);
					scale.pool <- c(scale.pool, scale.cursor );
					scale.cursor <- scale.cursor + 1;
				}
				
				if( length(col.pool) >= col.n || i==length(scaled.cols))
				{
					xtmp <- scale( x$data[, x$col.index[col.pool], drop=F], center = center[scale.pool], scale  = scale[scale.pool] );

					## NOTICE:
					## 
					## althogh r.dummy include 3 objects, but they are not duplicated.
					## Check by th follow methods:
					## lsos(envir=environment());
					## gc();
					## .Internal(inspect( r.dummy[[1]]) )
					## 
					## address = function(x) substring(capture.output(.Internal(inspect(x)))[1],2,17)
					## address(x$data)==address( r.dummy[[1]]) 
					## [1] TRUE

					#### x$data[,x$col.index[col.pool]] <- xtmp
					vec.cols <- x$col.index[col.pool];

					r.dummy  <- .C("bigmatrix_set_bycols", x$data, as.integer(NROW(x$data)), as.integer(NCOL(x$data)), as.integer(vec.cols), as.integer(NROW(vec.cols)), as.double(xtmp), NAOK = TRUE, DUP = FALSE, PACKAGE="Rgtsvm");
					col.pool <- c();
					scale.pool <- c();
				}

	        } 

	        invisible(return(x));
		}
	});

setGeneric("bigm.naction",
	def = function(x, na.action, nac=NULL){
		stopifnot(class(x) == "BigMatrix.refer");
		standardGeneric("bigm.naction");
	});

setMethod("bigm.naction", "BigMatrix.refer", function(x, na.action, nac=NULL)
	{
		if(is.null(nac)) 
		{
			x.rowsum <- na.action( rowSums(x$data) );;
			nac <- attr( x.rowsum, "na.action");
		}	
		
		if(!is.null(nac))
		{
			rm.rows <- x$row.index[ as.vector(nac) ];
			x$row.index <- c( x$row.index[ - as.vector(nac) ], rm.rows);
			x$row.subset <- NROW(x$row.index) - NROW(rm.rows);
		}
		
		x$nac[[1]] <- nac;
		return( nac );
	});


setGeneric("bigm.subset",
	def = function(x, cols=NULL, rows=NULL){
		stopifnot(class(x) == "BigMatrix.refer");
		standardGeneric("bigm.subset");
	});

setMethod("bigm.subset", "BigMatrix.refer", function(x,  cols=NULL, rows=NULL )
	{
		if(!is.null(cols)) 
		{
			subset <- c(1:x$col.subset)[cols];
			x$col.index <- c(x$col.index[subset], x$col.index[-subset]);
			x$col.subset <- length(subset);
		}	

		if(!is.null(rows)) 
		{
			subset <- c(1:x$row.subset)[rows];
			x$row.index <- c(x$row.index[subset], x$row.index[-subset]);
			x$row.subset <- length(subset);
		}	
	
        invisible(return(x));
	});

setGeneric("bigm.row.index",
	def = function(x, rows=NULL){
		stopifnot(class(x) == "BigMatrix.refer");
		standardGeneric("bigm.row.index");
	});

setMethod("bigm.row.index", "BigMatrix.refer", function(x, rows=NULL )
	{
		if(is.null(rows)) 
			rows <- c(1:x$row.subset);

		return( x$row.index[rows] );
	});
	
	
setGeneric("bigm.col.index",
	def = function(x, cols=NULL){
		stopifnot(class(x) == "BigMatrix.refer");
		standardGeneric("bigm.col.index");
	});

setMethod("bigm.col.index", "BigMatrix.refer", function(x, cols=NULL )
	{
		if(is.null(cols)) 
			cols <- c(1:x$col.subset);

		return( x$col.index[cols] );
	});
	

setGeneric("bigm.rbindcopy",
	def = function( x ){
		stopifnot(class(x) == "BigMatrix.refer");
		standardGeneric("bigm.rbindcopy");
	});

setMethod("bigm.rbindcopy", "BigMatrix.refer", function(x  )
	{
		new.rows <- c( x$row.index[1:x$row.subset], 
		   x$row.index[1:x$row.subset],
		   x$row.index[-(1:x$row.subset)] );
		
		x$row.index <- new.rows;
		x$row.subset <- x$row.subset*2;
		
        invisible(return(x));
	});
	
	
setGeneric("bigm.push",
	def = function(x, rds.save=FALSE){
		stopifnot(class(x) == "BigMatrix.refer");
		standardGeneric("bigm.push");
	});

setMethod("bigm.push", "BigMatrix.refer", function(x, rds.save=FALSE)
	{
		if(rds.save)
		{
			x$file.backup = tempfile(pattern="rgtsvm.bigm.", fileext=".rds");
			saveRDS(x$data, file = x$file.backup );
			cat("The training data is backuped in", x$file.backup, "\n");
	
			## system.time( saveRDS(2.7G data , "2.7Gdata.RDS" ))
			##   user  system elapsed 
			## 77.727   1.730  79.930
		}	
		
		hist <- list( file.backup = x$file.backup,
					row.index     = x$row.index,
					col.index     = x$col.index,
					row.subset    = x$row.subset,
					col.subset    = x$col.subset );

		x$stacks[[length(x$stacks) + 1]] <- hist;
		x$file.backup <- "";
		
		invisible(return(x));
	});

setGeneric("bigm.pop",
	def = function(x){
		stopifnot(class(x) == "BigMatrix.refer");
		standardGeneric("bigm.pop");
	});

setMethod("bigm.pop", "BigMatrix.refer", function(x)
	{
		bigm.restore( x );

		if( x$file.backup!="" )
		{
			unlink(x$file.backup);
			x$file.backup <- "";
		}	

		x$stacks[[length(x$stacks)]] <- NULL
		
		invisible(return(x));
	});


setGeneric("bigm.restore",
	def = function(x ){
		stopifnot(class(x) == "BigMatrix.refer");
		standardGeneric("bigm.restore");
	});

setMethod("bigm.restore", "BigMatrix.refer", function(x)
	{
		hist <- x$stacks[[ length(x$stacks) ]]

		if( hist$file.backup!="" )
		{
			x$data <- matrix(0, ncol=1, nrow=1);
			gc();
			x$data <- readRDS( hist$file.backup );
		}
		
		x$row.index     = hist$row.index;
		x$col.index     = hist$col.index;
		x$row.subset    = hist$row.subset;
		x$col.subset    = hist$col.subset;
		x$file.backup   = hist$file.backup;

		invisible(return(x));
	});


setGeneric("bigm.reshape",
	def = function(x){
		stopifnot(class(x) == "BigMatrix.refer");
		standardGeneric("bigm.reshape");
	});


reshape_cols<-function( x, cols.target )
{
	cols.current <- c(1:length(cols.target));
	
	move.size    <- floor( CACHE_SIZE/NROW(x$row.index) );
	move.current <- cols.current [ cols.target != cols.current ];
	move.target  <- cols.target  [ cols.target != cols.current ];
	
	if( length(move.target)==0 ) return;
	
	cursor <- 1;
	block.current <- c();
	block.target  <- c();
	
	while( cursor<=length(move.target) )
	{
		if ( move.target[cursor] == move.current[cursor] )
			next;

		if( length( intersect( c(block.current,  move.current[cursor] ), c(block.target,  move.target[cursor] ) ) )>0 || 
			length( block.current ) >= move.size )
		{
cat("COL:", block.target, "->", block.current, "\n");
			
			#temp <- x$data[,block.current,drop=F];
			#x$data[,block.current,drop=F] <- x$data[,block.target,drop=F];
			#x$data[,block.target,drop=F] <- temp;
			
			r.dummy  <- .C("bigmatrix_exchange_cols", x$data, 
				as.integer(NROW(x$row.index)), 
				as.integer(NROW(x$col.index)), 
				as.integer(block.current), 
				as.integer(NROW(block.current)), 
				as.integer(block.target), 
				as.integer(NROW(block.target)), 
				NAOK = TRUE, DUP = FALSE, PACKAGE="Rgtsvm");

			temp <- cols.current[block.current];
			cols.current[block.current] <- cols.target[block.target];
			cols.target[block.target] <- temp;
			
			block.target  <- c();
			block.current <- c();
		}
		
		block.target <-  c(block.target,  move.target[cursor] );
		block.current <- c(block.current, move.current[cursor] );
		
		cursor <- cursor +1;
	}
	
	if (any(cols.current != cols.target)) error("stop to move");
	
}


## e.gi.
##  
##   current: 1 2 3 4 5 6 7 8 9
##   target : 9 8 4 3 5 6 1 2 7  
##
##   Step1:   9 8           2 1 
##       2:   9 8 4 3       2 1 
##       3:   9 8 4 3 5 6 7
##       4:   9 8 4 3 5 6 1   7
reshape_rows<-function( x, rows.target )
{
	rows.current <- c(1:length(rows.target));
	
	move.size    <- floor( CACHE_SIZE/NROW(x$col.index) );
	move.current <- rows.current [ rows.target != rows.current ];
	move.target  <- rows.target  [ rows.target != rows.current ];
	
	if( length(move.target)==0 ) return;
	
	cursor <- 1;
	block.current <- c();
	block.target  <- c();
	
	while( cursor<=length(move.target) )
	{
		if ( move.target[cursor] == move.current[cursor] )
			next;

		if( length( intersect( c(block.current,  move.current[cursor] ), c(block.target,  move.target[cursor] ) ) )>0 || 
			length( block.current ) >= move.size )
		{
cat("ROW:", block.target, "->", block.current, "\n");

			# temp <- x$data[block.current,,drop=F];
			# x$data[block.current,,drop=F] <- x$data[block.target,,drop=F];
			# x$data[block.target,,drop=F] <- temp;

			r.dummy  <- .C("bigmatrix_exchange_rows", x$data, 
				as.integer(NROW(x$row.index)), 
				as.integer(NROW(x$col.index)), 
				as.integer(block.current), 
				as.integer(NROW(block.current)), 
				as.integer(block.target), 
				as.integer(NROW(block.target)), 
				NAOK = TRUE, DUP = FALSE, PACKAGE="Rgtsvm");

			temp <- rows.current[block.current];
			rows.current[block.current] <- rows.target[block.target];
			rows.target[block.target] <- temp;
			
			block.target  <- c();
			block.current <- c();
		}
		
		block.target <-  c(block.target,  move.target[cursor] );
		block.current <- c(block.current, move.current[cursor] );
		
		cursor <- cursor +1;
	}
	
	if (any(row.current != rows.target)) error("stop to move");
	
}


setMethod("bigm.reshape", "BigMatrix.refer", function(x)
	{
		reshape_cols(x, x$col.index );
		reshape_rows(x, x$row.index );

		x$file.backup  = "";
		x$row.index    = c(1:nrow(x$data));
		x$col.index    = c(1:col(x$data));
		
		invisible();
	});


attach.bigmatrix <- function( data )
{
	BigMatrix.refer <- getRefClass("BigMatrix.refer");
	bigm.x <- BigMatrix.refer(data = data, 
			file.rdata    = "", 
			variable      = "",
			scale         = list(),
			nac           = list(),
			stacks        = list(),
			file.backup   = "",				## SAVE in the stacks when push or pop 
			row.index     = 1:NROW(data),	## SAVE in the stacks when push or pop 
			col.index     = 1:NCOL(data),	## SAVE in the stacks when push or pop
			row.subset    = NROW(data),		## SAVE in the stacks when push or pop
			col.subset    = NCOL(data))		## SAVE in the stacks when push or pop
	
	return(bigm.x);		
}

load.bigmatrix <- function(file.data, variable=NULL)
{
	BigMatrix.refer <- getRefClass("BigMatrix.refer");

	x <- BigMatrix.refer(data = matrix(1,ncol=1, nrow=1), 
			file.rdata    = file.data, 
			variable      = "",
			scale         = list(),
			nac           = list(),
			stacks        = list(),
			file.backup   = "",	
			row.index     = 1,
			col.index     = 1,
			row.subset    = 1,
			col.subset    = 1)

	
	if( is.null(variable)  )
	{
		x$data <- readRDS(file.data);
		x$variable = "";
	}	
	else
	{
		load( file.data );
		eval( parse( text=paste( "x$data=", variable )));
		x$variable      = variable;
		gc();
	}

	x$row.index     = 1:NROW(x$data);
	x$col.index     = 1:NCOL(x$data);
	x$row.subset    = NROW(x$data);
	x$col.subset    = NCOL(x$data);

	return(x);		
}

bigm.reload <- function( x )
{
	if (x$file.backup != "" ) 
		unlink(x$file.backup);

	if( x$variable == "" )
	{
		x$data <- readRDS(x$file.rds)
	}
	else
	{
		load(x$file.rds);
		x$data <- matrix(0, nrow=1, ncol=1);
		eval(parse(text=paste( "x$data=", x$variable )));
		gc();
	}
		
	x$scale         = list();
	x$nac           = list();
	x$stacks        = list();
	x$file.backup   = "";
	x$row.index     = 1:NROW(x$data);
	x$col.index     = 1:NCOL(x$data);
	x$row.subset    = NROW(x$data);
	x$col.subset    = NCOL(x$data);
	
	return(x);		
}
