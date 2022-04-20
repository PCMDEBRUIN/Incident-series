library('coda')
library('invgamma')
library("tidyverse")
library("writexl")

T = 10000 # size of Markov chain
m = 3000 # burn in period
n = 27 * 3 # number of nurses
r = 201 # number of shifts (Lucia)

# parameter choice of Gill et al.
mu0 <- 27/1734
rho0 <- 1 
N0 <- 1 - pnbinom(13, size = rho0, prob = 1 / (1 + r * mu0 / rho0), log = FALSE)

# simulate data based on Gill's model
x <- rnbinom(n, size = rho0, prob = 1 / (1 + r * mu0 / rho0)) 
# number of incidents that a nurse experiences during r shifts
x_bar <- mean(x)
x2_bar <- mean(x ** 2)

# parameters of interest
rho <- rep(0, T + 1)
mu <- rep(0, T + 1)

# quantity of interest P(N(t) >= 14)
N <- rep(0, T + 1)

# initialization
rho[1] = 1
mu[1] = 0.015
N[1] = 0

# Gibbs sampler
for(t in 1:T){
  # lambda
  lambda <- rep(0, n)
  for(i in 1:n){
    lambda[i] <- rgamma(1, shape = x[i] + rho[t], rate = r + rho[t] / mu[t])
  }
  
  # full conditional mu (inverse gamma)
  mu[t + 1] <- rinvgamma(1, n * rho[t] - 0.01, rho[t] * sum(lambda))
  
  # full conditional rho (discretized)
  prob <- rep(0, 300)
  for(j in 1:300){
    # p0 is Gamma(1,1) distributed
    prob[j] <- (dgamma(j * 0.005, shape = 1, rate = 1, log = FALSE) 
                / (gamma(j * 0.005) ** n)) * exp(-(j * 0.005 / mu[t + 1])
                                                 * sum(lambda) + n * j * 0.005 * log(j * 0.005 / mu[t + 1]) 
                                                 + (j * 0.005 - 1) * sum(log(lambda)))
  }
  c <- sum(prob)
  rho[t + 1] <- sample((1:300) * 0.005, size = 1, replace = TRUE, prob = prob / c)
  
  # probability experience at least 14 incidents
  N[t + 1] <- 1 - pnbinom(13, size = rho[t + 1], 
                          prob = 1 / (1 + r * mu[t + 1] / rho[t + 1]), log = FALSE)
}

# save dataset
save(x, rho, mu, N, file = "Gibbs_sampler_dataset.Rdata")
df <- data.frame("rho" = rho, "mu" = mu, "probability" = N)
dg <- data.frame("incidents" = x)
write_xlsx(dg,"~/Documents/Mathematical Sciences/Research Thesis/R/Plot/Gibbs sampler/
simulated_dataset_incidents_Gibbs.xlsx")
write_xlsx(df,"~/Documents/Mathematical Sciences/Research Thesis/R/Plot/Gibbs sampler/
simulated_dataset_outcome_Gibbs.xlsx")

setwd('/Users/patricia/Documents/Mathematical Sciences/Research Thesis/R/Plot/
Gibbs sampler')

# traceplot rho
pdf("GS_rho.pdf", width = 10, height = 6)
traceplot(as.mcmc(rho), main = expression(paste("Traceplot of ", hat(rho))), 
          ylab = expression(paste(rho)), xaxt = "n")
axis(1, at = seq(0, T, by = T / 5))
abline(h = rho0, col = 'red4', lwd = 3, lty = 3)
abline(h = mean(rho[(m + 1):(T + 1)]), col = 'aquamarine4', lwd = 3, lty = 3)
legend("bottomleft", inset = 0.025, legend = c("True value", "Mean estimated value"),
       col = c("red4", "aquamarine4"), lty = 3:3, cex = 0.6, lwd = 3:3, bg = 'white')
dev.off()

# traceplot mu
pdf("GS_mu.pdf", width = 10, height = 6)
traceplot(as.mcmc(mu), main = expression(paste("Traceplot of ", hat(mu))), 
          ylab = expression(paste(mu)), xaxt = "n")
axis(1, at = seq(0, T, by = T / 5))
abline(h = mu0, col = 'red4', lwd = 3, lty = 3)
abline(h = mean(mu[(m + 1):(T + 1)]), col = 'aquamarine4', lwd = 3, lty = 3)
legend("bottomleft", inset = 0.025, legend = c("True value", "Mean estimated value"),
       col = c("red4", "aquamarine4"), lty = 3:3, cex = 0.6, lwd = 3:3, bg = 'white')
dev.off()

# traceplot P(N(t) >= 14)
pdf("GS_N.pdf", width = 10, height = 6)
traceplot(as.mcmc(N), main = expression(paste("Traceplot of ", P(N(t) >= 14))), 
          ylab = "N", xaxt = "n")
axis(1, at = seq(0, T, by = T / 5))
abline(h = N0, col = 'red4', lwd = 3, lty = 3)
abline(h = mean(N[(m + 1):(T + 1)]), col = 'aquamarine4', lwd = 3, lty = 3)
legend("topright", inset = 0.025, legend = c("True value", "Mean estimated value"),
       col = c("red4", "aquamarine4"), lty = 3:3, cex = 0.6, lwd = 3:3, bg = 'white')
dev.off()

# density plot rho
pdf("GS_rho_plot.pdf", width = 10, height = 6)
hist(rho[(m + 1):(T + 1)], breaks = 100,
     main = expression(paste("Posterior density of ", hat(rho))),
     xlab = expression(paste(rho)),
     freq = FALSE)
lines(density(rho[(m + 1):(T + 1)]), col = "red4", lwd = 2) # add corresponding density
dev.off()

# density plot mu
pdf("GS_mu_plot.pdf", width = 10, height = 6)
hist(mu[(m + 1):(T + 1)], breaks = 100,
     main = expression(paste("Posterior density of ", hat(mu))),
     xlab = expression(paste(mu)),
     freq = FALSE)
lines(density(mu[(m + 1):(T + 1)]), col = "red4", lwd = 2) # add corresponding density
dev.off()

# summary true value, mean and sd of parameters and P(N(t) >= 14) 
par <- data.frame(N[(m + 1):(T + 1)], rho[(m + 1):(T + 1)], mu[(m + 1):(T + 1)])
colnames(par) <- c('P(N(t) >= 14)', 'rho', 'mu')
trueval <- c(N0, rho0, mu0)
mean <- sapply(par, FUN = mean)
sd <- sapply(par, FUN = sd)
dh <- cbind(trueval, mean, sd)
dh

# save table with outcomes
save(dh, file = "GS_outcomes_table.Rdata")