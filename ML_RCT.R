##########################################################################################################
##
## Example implementation of machine learning assisted adjustment for randomized controlled trials
## Date: April 6, 2025
## Citation: Yu, H., Hutson, A.D. and Ma, X., 2024. Machine Learning Assisted Adjustment Boosts Efficiency of Exact Inference in Randomized Controlled Trials. arXiv preprint arXiv:2403.03058.
##
##########################################################################################################

require(mvtnorm)
require(randomForest)

## Generate a synthetic data

set.seed(123)
p <- 40
n <- 100
betas <- rep(0.8, 4)
Sigma <- matrix(0, nrow = p, ncol = p)
diag(Sigma) <- 1
sigmoid <- function(x) {y <- exp(x)/(1+exp(x)); return(y)}
x <- as.data.frame(rmvnorm(n, mean = rep(0, p), sigma = Sigma))
x_mat <- as.matrix(x)
arm <- sample(c(rep(0, n/2), rep(1, n/2)))
y <- sigmoid(x_mat[,1]*0.5) * betas[1] + x_mat[,2]^2 * betas[2] + cos(x_mat[,3]) * betas[3] + x_mat[,4] * betas[4] + rnorm(n)
y[arm==1] <- y[arm==1] + 1
df <- data.frame(y, x_mat)

## Build random forest model
rf <- randomForest(y~., df, ntree = 500)

## Obtain OOB predictions
pred <- rf$predicted

## Calculate residuals
r <- y-pred
r0 <- r[arm == 0]
r1 <- r[arm == 1]

## Wilcoxon rank-sum test on residuals
## The p-value, difference in location, and 95% confidence interval can be obtained
w_test <- wilcox.test(r1, r0, exact = TRUE, conf.int = TRUE)
