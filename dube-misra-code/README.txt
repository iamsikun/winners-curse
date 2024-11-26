#=========================================================================#
# ZIPRECRUITER PROJECT
#
#  CODE LIBRARY FOR REPLICATION
#
# JP Dube and Sanjog Misra, 12-18-2021
#
#=========================================================================#

Directory system requirements:
 - all raw data and processed data in /data
 - all estimates in /output
 - all figures in /figures

-----------

1) Run mksimdata.R
	--> create a synthetic dataset
	--> store estimation data in /data
	--> store "true" coefficients in /data

2) Run DemandEstimation.R
	--> estimate MLE Binary Logit
	--> estimate GAMLR Binary Logit
	--> store estimates in /output
	--> analyze posterior heterogeneity

3) Run Pricing.R
	--> decision-theoretic uniform pricing
	--> decision-theoretic personalized pricing
	--> Evaluate Welfare
