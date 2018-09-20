## NOTE: In process of making installation easier, as follows:
##       The bigWig package provides all required Kent source dependencies,
##       and should be appearing in CRAN soon.
#export R_LIBS

.PHONY: Rgtsvm

R_dependencies:
	@echo "Installing R dependencies" # to:" ${R_LIBS}
	#mkdir -p ${R_LIBS}
	R --no-save < rDeps.R


