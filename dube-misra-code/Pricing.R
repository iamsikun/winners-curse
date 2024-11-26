#=========================================================================#
#  OPTIMIZED UNIFORM AND PERSONALIZED PRICING
#  JP Dube and Sanjog Misra
#
#  12/18/2021
#=========================================================================#

#---------+---------+---------+---------+---------+---------+
# settings
#---------+---------+---------+---------+---------+---------+
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
library(gamlr)
library(sfsmisc)

set.seed(1)

#---------+---------+---------+---------+---------+---------+
# FUNCTIONS	
#---------+---------+---------+---------+---------+---------+
# Copy tables to clipboard (for pasting)
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

share=function(p,aa,bb)
{
  share = 1/(1+exp(-aa-bb*p))
  return(share)
}

profvec = function (p0,aa,bb) { p0*mean(share(p0,aa,bb),na.rm=T) }
profvec = Vectorize(profvec,"p0")


profit = function (p0,aa,bb) 
{
  if (ncol(aa)==1) {
    S = mean(share(p0,aa,bb),na.rm=T)
  } else {
    S = mean( apply( share(p0,aa,bb), 2 , mean,na.rm=T) )
  }
  prof = p0*S
  return(prof)
}


# RUN TARGETED PRICING
runTarg = function (a,b) 
{
  pstar = matrix(0,nrow(a),1)
  NBtemp = dim(b)[2]
  for (rr in 1:nrow(a)) {
    aa = as.matrix(a[rr,])
    bb = as.matrix(b[rr,])
    ptemp = optimize(f=profit,c(0,10),aa=aa,bb=bb,maximum=T)
    pstar[rr] = ptemp$maximum
  }
  pp = pstar%*%matrix(1,1,NBtemp)
  S = share(pp,a,b)
  Sstar = apply( S , 2 , mean,na.rm=T)
  profstar = apply( S*pp , 2 , mean,na.rm=T)
  CSstar = apply(-log(1+exp( a+b*pp))/b,2,mean,na.rm=T)
  S = share(pp,a,b)
  Sstartrue = apply( S , 2 , mean,na.rm=T)
  profstartrue = apply( S*pp , 2 , mean,na.rm=T)
  CSstartrue = apply(-log(1+exp( a+b*pp))/b,2,mean,na.rm=T)
  return(list(pstar=pstar,Sstar=Sstar,profstar=profstar,CSstar=CSstar,Sstartrue=Sstartrue,profstartrue=profstartrue,CSstartrue=CSstartrue))
}

# RUN Uniform PRICING
runUnif = function (a,b) 
{
  out = optimize(f=profit,c(0,10),aa=a,bb=b,maximum=T)
  #out = optim(fn=profit,gr=profitFOC,par=0,aa=a,bb=b,control=list(fnscale=-1,maxit=1000), method = "BFGS")
  punif = out$maximum
  Sunif = apply( share(punif,a,b) , 2 , mean,na.rm=T)
  profunif = Sunif*punif
  CSunif = apply(-log(1+exp( a+b*punif))/b,2,mean,na.rm=T)
  Suniftrue = apply( share(punif,a,b) , 2 , mean,na.rm=T)
  profuniftrue = Suniftrue*punif
  CSuniftrue = apply(-log(1+exp( a+b*punif))/b,2,mean,na.rm=T)
  return(list(punif=punif,Sunif=Sunif,profunif=profunif,CSunif=CSunif,Suniftrue=Suniftrue,profuniftrue=profuniftrue,CSuniftrue=CSuniftrue))
}


# Consumer Surplus
CS = function(V,r){
  if(r==0){
    CS = exp( apply( log(V) ,2,mean,na.rm=T) )
  } else {
    CS = (apply( (V)^r ,2,mean,na.rm=T))^(1/r)
  }
  return(CS)
}


# Change in Consumer Surplus
Delta_CS = function(r0,Vunif,Vstar){
  CS_unif = CS(Vunif,r=r0)
  CS_PD = CS(Vstar,r=r0)
  H = mean( CS_unif-CS_PD, na.rm=T )
  return(H)
}


#---------+---------+---------+---------+---------+---------+
# LOAD DATA
#---------+---------+---------+---------+---------+---------+
load(file="data/estimdata.Rdata")

#### List of Relevant Variables
# List of Relevant Variables
vars = colnames(datz)[2:(ncol(datz)-1)]
X = datz[,vars]
P = datz[,"P"]
mm = as.matrix(cbind(X,P,X*(P%*%matrix(1,1,ncol(X)))))
colnames(mm)[(dim(X)[2]+2):(2*dim(X)[2]+1)] = paste("P:",colnames(mm)[(dim(X)[2]+2):(2*dim(X)[2]+1)],sep="")
mm.base = as.matrix(X)
yy = factor(datz$FT)
yy2 = factor(datz$Y)
N = nrow(mm)


#---------+---------+---------+---------+---------+---------+
# LOAD DEMAND ESTIMATES
#---------+---------+---------+---------+---------+---------+
load("output/gmcfs.Rdata")
NB = dim(gm.cfs)[2] 		                    # Number of Bootstraps
X = cbind(matrix(1,N,1),mm)
colnames(X)[1] = "intercept"
intmn_vars = grep(":",colnames(X))          # interaction variables
priceind = which(colnames(X)=="P")
mainmn = X[,-intmn_vars]
intmn = X[,intmn_vars]/X[,priceind]

a = (mainmn[,-priceind]%*%gm.cfs[-c(intmn_vars,priceind),])
b = kronecker(matrix(gm.cfs[priceind,],nrow=1),matrix(1,N,1)) + intmn%*%gm.cfs[intmn_vars,]
# trim positive-valued draws of b
bb = sort(b)
cutoff = min(which(bb==min(bb[bb>0])))/length(bb)-.01
b[b>quantile(b,prob=cutoff)] = quantile(b,prob=cutoff)
b[b>0] = NaN


#---------+---------+---------+---------+---------+---------+
# Pricing Simulations
#---------+---------+---------+---------+---------+---------+

# personalized pricing
outtarg = runTarg(a,b)
oo = matrix(1,1,100)

# uniform pricing
outunif = runUnif(a,b)

# Output results
save(outtarg,outunif,file="output/pricesimulations.Rdata")

#---------+---------+---------+---------+---------+---------+
# Profitability
#---------+---------+---------+---------+---------+---------+
punif = outunif$punif
profunif = outunif$profuniftrue
Sunif = outunif$Sunif

pstar = outtarg$pstar
SPD = apply( 1-1/(1+exp(a)) , 2, mean , na.rm=T)
profPD = apply( (1-1/(1+exp(a)) )*(-a/b - 1/b*(-a+(1+exp(a))*log(1+exp(a))/exp(a))) , 2 , mean, na.rm=T )
profstar = outtarg$profstartrue

Dprof = mean(profstar-profunif)/mean(profunif)*100
print(paste("Expected Profit Increase (personalized vs unif):" , round(Dprof,2), "%"))


#---------+---------+---------+---------+---------+---------+
# Consumer Surplus
#---------+---------+---------+---------+---------+---------+
CSunif = outunif$CSuniftrue
CSstar = outtarg$CSstartrue


#---------+---------+---------+---------+---------+---------+
# Bergemann, Brooks and Morris (AER 2015) Surplus Triangle Graph
#---------+---------+---------+---------+---------+---------+
A = matrix(c(mean(CSunif),mean(profunif)),nrow=1)
B = matrix(c(0,mean(profPD)),nrow=1)
C = matrix(c(mean(profPD)-mean(profunif),mean(profunif)),nrow=1)
D = matrix(c(0,mean(profunif)),nrow=1)

E0 = matrix(c(mean(CSstar),mean(profstar)),nrow=1)
E = matrix(c(mean(CSstar),mean(profstar)),nrow=1)

pdf(file="figures/Surplustriangle.pdf")
plot(c(0,C[1]*1.02),c(0,B[2]*1.02),type="n",main="The Surplus Triangle",xlab="Posterior Expected Customer Surplus",ylab="Posterior Expected Profit",cex.lab=1.5,cex.main=1.5,cex.axis=1.5)
polygon(x=rbind(B[1],C[1],D[1]),y=rbind(B[2],C[2],D[2]),density=NA,col="lightgray",border="black")
points(rbind(B,C,D),cex=2,pch=16)
points(x=A[1],y=A[2],col="darkgreen",cex=2,pch=16)
points(E,col="red",cex=1.4,pch=16)
text(rbind(C,D),c("C","D"),pos=1,cex=1.5)
text(B,"B",pos=3,cex=1.5)
text(A,c("A"),pos=1,col="darkgreen",cex=1.5)
text(E*c(1.001,1.05),"E",pos=4,col="red",offset=.3,cex=1.5)
text(rbind(C,D),c("C","D"),pos=1,cex=1.5)
text(B,"B",pos=3,cex=1.5)
text(A,c("A"),pos=1,col="darkgreen",cex=1.5)
text(E*c(1.001,1.05),"E",pos=4,col="red",offset=.3,cex=1.5)
legend("topright",legend=c(expression("A: P"[unif]),"B: expected perfect PD",expression("C: CS-max & P"[unif]),expression("D: CS-min & P"[unif]),"E: Personalized Pricing (full)")
       ,pch=19,col=c("darkgreen","black","black","black","red"),text.col=c("darkgreen","black","black","black","red"),cex=1)
dev.off()


#---------+---------+---------+---------+---------+---------+
# Social Welfare Functions
#---------+---------+---------+---------+---------+---------+
# (1) individual posterior expected surplus
Vunif = as.matrix(apply( -log(1+exp(a+b*punif))/b , 1,mean))
Vstar = as.matrix(apply( -log(1+exp(a+b*(pstar%*%matrix(1,1,ncol(a)))))/b , 1,mean))

# (2) R=1
CS_r1_unif = CS(Vunif,1)
CS_r1_star = CS(Vstar,1)
print(paste("Welfare difference (unif vs personalized) when r = 1:" , round(Delta_CS(1,Vunif,Vstar),2)))

# (3) R=0
CS_r0_unif = CS(Vunif,0)
CS_r0_star = CS(Vstar,0)
print(paste("Welfare difference (unif vs personalized) when r = 0:" , round(Delta_CS(0,Vunif,Vstar),2)))

# (4) R=-1
CS_rn1_unif = CS(Vunif,-1)
CS_rn1_star = CS(Vstar,-1)
print(paste("Welfare difference (unif vs personalized) when r = -1:" , round(Delta_CS(-1,Vunif,Vstar),2)))
