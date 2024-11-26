#=========================================================================#
#  DEMAND ESTIMATION
#  JP Dube and Sanjog Misra
#
#  12/18/2021
#=========================================================================#

#---------+---------+---------+---------+---------+---------+
# settings
#---------+---------+---------+---------+---------+---------+
NB = 100 		# Number of Bootstraps 

if(.Platform$OS.type != "unix") {
  setwd("C:/Users/jdube/Dropbox/ZipProject/code/JPE replication package/")
}
if(.Platform$OS.type == "unix") {
  setwd("~/Dropbox/ZipProject/code/JPE replication package/")
}


#---------+---------+---------+---------+---------+---------+
# Preliminaries
#---------+---------+---------+---------+---------+---------+
library(Rcpp)
library(readxl)
library(glmnet)
library(Hmisc)
library(gamlr)


#---------+---------+---------+---------+---------+---------+
# FUNCTIONS	#
#---------+---------+---------+---------+---------+---------+
copy.table <- function(obj, size = 4096) {
  clip <- paste('clipboard-', size, sep = '')
  f <- file(description = clip, open = 'w')
  write.table(obj, f, row.names = FALSE, sep = '\t')
  close(f)  
}

# Credibility interval
mkcred = function(x,prob) {
  return( paste("(",round(quantile(x,(1-prob)/2),2),",",round(quantile(x,1-(1-prob)/2),2),")",sep="") )
}


#---------+---------+---------+---------+---------+---------+
# LOAD DATA
#---------+---------+---------+---------+---------+---------+
load(file="data/estimdata.Rdata")


#---------+---------+---------+---------+---------+---------+
#  DEMAND ESTIMATION
#---------+---------+---------+---------+---------+---------+
# List of Relevant Variables
vars = colnames(datz)[2:(ncol(datz)-1)]
X = datz[,vars]
P = datz[,"P"]
mm = as.matrix(cbind(X,P,X*(P%*%matrix(1,1,ncol(X)))))
colnames(mm)[(dim(X)[2]+2):(2*dim(X)[2]+1)] = paste("P:",colnames(mm)[(dim(X)[2]+2):(2*dim(X)[2]+1)],sep="")
yy = factor(datz$FT)
yy2 = factor(datz$Y)
N = nrow(mm)

# (1) LOGIT VIA MLE
mle.out = glm(yy2~mm,family='binomial')
mle.aic = mle.out$aic
mle.bic = mle.out$deviance + log(length(yy2))*length(mle.out$coefficients)

# (2) Weighted Likelihood Bootstrap GAMLR
y=as.numeric(yy2)-1 	#  Gamlr requires -1/+1 coding
Kp = ncol(mm) 		# Dimension of Variables (in design matrix)
K = 0

# (2a) FIRST RUN GAMLR
set.seed(2)
gamlr.out = cv.gamlr(mm,y,family="binomial",nfold=5)
gamlr.bic = (gamlr.out$gaml$deviance + gamlr.out$gamlr$df*log(gamlr.out$gamlr$nobs))[gamlr.out$seg.min] 
prior.od = 1.8 # prior over dispersion parameter (researcher's choice)
wts = rexp(N)^prior.od							                # Rubin's Dirichlet Weighting approach 
idx=sample(1:N,N,prob=wts,replace=TRUE) 				# Sample based on Dirichlet Weights
gamlr.out.tmp = cv.gamlr(mm[idx,],y,family="binomial",nfold=5)

# (2b) SECOND RUN GAMLR BOOTSTRAP
glm.out = gmd.out =  gm.out=list()
set.seed(2)

for(b in 1:NB)
{
  wts = rexp(N)^prior.od 							              # Rubin's Dirichlet Weighting approach
  idx = sample(1:N,N,prob=wts,replace=TRUE) 	  # Sample based on Dirichlet Weights
  gm.out[[b]] = cv.gamlr(mm[idx,],y[idx],family="binomial",nfold=5) 	# Run Cross Validated Gamlr
  gm.set = which(coef(gm.out[[b]])!=0)  				# Active Set Index
  gm.set = gm.set[2:length(gm.set)]-1  					# Active Set
  gmd.out[[b]] = glm(y[idx]~mm[idx,gm.set],family='binomial') 		    # Post Selection GLM
  cat("Bootstrap ",b," completed.\n")
}

# (2c) THIRD RUN POST-GAMLR MLE
gm.set = which(coef(gamlr.out)!=0)  				  # Active Set Index
gm.set = gm.set[2:length(gm.set)]-1  					# Active Set
postLSmle.out = glm(y~mm[,gm.set],family='binomial')
postLSmle.bic = postLSmle.out$deviance + log(length(yy2))*length(postLSmle.out$coefficients)

# Save GAMLR and post-GAMLR MLE output
save(gm.out,mle.out,gamlr.out,postLSmle.out,gm.set,gm.out,mle.bic,gamlr.bic,postLSmle.bic,file="output/gmoutput.Rdata")


#---------+---------+---------+---------+---------+---------+
# Retain Coefficients based on Minimum Lambda
# From Gamlr
#---------+---------+---------+---------+---------+---------+
gm.cfs = matrix(0,nrow(coef(gm.out[[1]])),NB)
for (rr in 1:NB) {
	bb = as.matrix(coef(gm.out[[rr]]))
	gm.cfs[,rr] = bb
}
barplot(rowSums(gm.cfs!=0)/NB)

# Compute BICs
gm.df = lapply(gm.out, function(x) x$gamlr$df)
gm.bic = sapply(gm.out,function(x) x$gamlr$deviance+x$gamlr$df*log(x$gamlr$nobs) )[sapply(gm.out,function(x) x$seg.min),cbind(as.matrix(1:NB))]

# SAVE Coeffs (gm.cfs)
save(gm.cfs,gm.df,gm.bic,vars,file="output/gmcfs.Rdata")


#---------+---------+---------+---------+---------+---------+
# COMPARE MLE (unrestricted) and GAMLR
#---------+---------+---------+---------+---------+---------+
outputBIC = as.matrix(c( round(mle.bic,2), round(postLSmle.bic,2) , round(gamlr.bic,2) , mkcred(gm.bic,1)))
rownames(outputBIC) = c("MLE","postLS MLE","GL","WLB")
copy.table(outputBIC)
print(outputBIC)

length(which(coef(gamlr.out)!=0))
range(sapply(gm.out,function(x) length(which(coef(x)!=0))))


#---------+---------+---------+---------+---------+---------+
# POSTERIOR HETEROGENEITY
#---------+---------+---------+---------+---------+---------+
load("output/gmcfs.Rdata")
load("data/trueprefs.Rdata")

X = cbind(matrix(1,N,1),mm)
colnames(X)[1] = "intercept"
intmn_vars = grep(":",colnames(X))                         # interaction variables
priceind = which(colnames(X)=="P")
mainmn = X[,-intmn_vars]
intmn = X[,intmn_vars]/X[,priceind]
a = (mainmn[,-priceind]%*%gm.cfs[-c(intmn_vars,priceind),])
b = kronecker(matrix(gm.cfs[priceind,],nrow=1),matrix(1,N,1)) + intmn%*%gm.cfs[intmn_vars,]
keep = b<0
a[keep==0] = NaN
b[keep==0] = NaN

# Compute posterior mean WTP at mean price
EWTP = apply(-log(1+exp( a+b*mean(X[,"P"])))/b,1,mean,na.rm=T)
quantile(EWTP,probs=c(.025,.5,.975))

# Compute distribution of posterior mean price coefficients
Eb = apply(b,1,mean,na.rm=T)
quantile(Eb,probs=c(.025,.5,.975))

# Plot comparing "true" and "estimated" price coefficients
pdf(file="figures/heterog.pdf")
par(mfrow=c(2,1))
hist(apply(b,1,mean),col='grey',xlim=c(min(btrue),max(btrue)),xlab = expression(paste("E[",beta(x[i]),"|D,",x[i],"]")),main="Panel (a): Posterior Mean Price Coefficient",breaks=100)
hist(apply(btrue,1,mean),col='grey',xlim=c(min(btrue),max(btrue)),xlab = expression(paste(beta(x[i]))),main="Panel (b): True Coefficient",breaks=100)
dev.off()
print(paste("correlation between E(b(x)|data) and true beta(x):" ,toString(round(cor(cbind(apply(b,1,mean,na.rm=T),btrue))[2],3)) ))

