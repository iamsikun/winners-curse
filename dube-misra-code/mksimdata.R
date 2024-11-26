#=========================================================================#
#  Simulate Choice Data
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
  setwd("~/ZipProject/code/JPE replication package/")
}


#---------+---------+---------+---------+---------+---------+
# FUNCTIONS	
#---------+---------+---------+---------+---------+---------+
share=function(p,aa,bb)
{
  share = 1/(1+exp(-aa-bb*p))
  return(share)
}


#---------+---------+---------+---------+---------+---------+
# PRELIMINARIES
#---------+---------+---------+---------+---------+---------+
set.seed(1) 

N = 8000
Nx = 133


#---------+---------+---------+---------+---------+---------+
# Preferences
#---------+---------+---------+---------+---------+---------+
betatrue = matrix(runif(Nx*2),ncol=2)
betatrue[,1] = betatrue[,1]/2.5
betatrue[,2] = -abs(betatrue[,2])*2.5
max(betatrue[betatrue[,2]<0,2])
# randomly select which features have non-zero weights
betatrue[-sample(1:Nx,30),1] = 0
betatrue[-sample(1:Nx,30),2] = 0
CholcovX = diag(Nx)+as.matrix(tril(matrix(runif(Nx*Nx),nrow=Nx)))
X = abs(matrix(rnorm(N*Nx),nrow=N)%*%CholcovX)
X = X/max(X)      # scale to lie in [-1,1]
P = as.matrix(runif(N)/2)
atrue = X%*%betatrue[,1]
btrue = X%*%betatrue[,2]
Prob = 1/(1+exp(-btrue*P-atrue))
max(Prob)
min(Prob)
mean(Prob)
median(Prob)

elastrue = btrue*(1-share(P,atrue,btrue))*P
quantile(elastrue,c(.025,.5,.975))


#---------+---------+---------+---------+---------+---------+
# Choices
#---------+---------+---------+---------+---------+---------+
Y = matrix(0,nrow=N,ncol=1)
oo = as.matrix(c(1,0))
for(ii in 1:N){Y[ii]=matrix(rmultinom(1,1,prob=as.matrix(c(Prob[ii],1-Prob[ii]))),nrow=1)%*%oo}


#---------+---------+---------+---------+---------+---------+
# Save Data
#---------+---------+---------+---------+---------+---------+
datz = data.frame(Y,X,P)
save(datz,file="data/estimdata.Rdata")
save(atrue,btrue,betatrue,elastrue,file="data/trueprefs.Rdata")
