# Generate reference values for cross-validation against stats-cli
# Run: Rscript generate_reference.R

library(jsonlite)
library(car)
library(nortest)
library(effectsize)
library(rstatix)
library(DescTools)

set.seed(42)
output <- list()

# 1. Descriptive Statistics
cat("1. Descriptive Statistics\n")
data(iris)
setosa_sl <- iris$Sepal.Length[iris$Species == "setosa"]
output$descriptive_iris_setosa <- list(
  n = length(setosa_sl),
  mean = mean(setosa_sl),
  sd = sd(setosa_sl),
  median = median(setosa_sl),
  min = min(setosa_sl),
  max = max(setosa_sl),
  q1 = quantile(setosa_sl, 0.25, names=FALSE),
  q3 = quantile(setosa_sl, 0.75, names=FALSE),
  iqr = IQR(setosa_sl),
  skewness = as.numeric(e1071::skewness(setosa_sl, type=2)),
  kurtosis = as.numeric(e1071::kurtosis(setosa_sl, type=2))
)

# 2. Normality Tests
cat("2. Normality Tests\n")
shapiro <- shapiro.test(setosa_sl)
lillie <- lillie.test(setosa_sl)
output$normality_iris_setosa <- list(
  shapiro_w = as.numeric(shapiro$statistic),
  shapiro_p = shapiro$p.value,
  lilliefors_d = as.numeric(lillie$statistic),
  lilliefors_p = lillie$p.value
)

# 3. One-sample t-test
cat("3. One-sample t-test\n")
t_one <- t.test(setosa_sl, mu = 5.0)
cohens_d_one <- effectsize::cohens_d(setosa_sl, mu = 5.0)
output$ttest_one_sample <- list(
  t_stat = as.numeric(t_one$statistic),
  df = as.numeric(t_one$parameter),
  p_value = t_one$p.value,
  ci_lower = t_one$conf.int[1],
  ci_upper = t_one$conf.int[2],
  mean = mean(setosa_sl),
  se = sd(setosa_sl) / sqrt(length(setosa_sl)),
  cohens_d = as.numeric(cohens_d_one)
)

# 4. Two-sample t-test
cat("4. Two-sample t-test\n")
versi_sl <- iris$Sepal.Length[iris$Species == "versicolor"]
t_welch <- t.test(setosa_sl, versi_sl)
cohens_d_two <- effectsize::cohens_d(setosa_sl, versi_sl)
output$ttest_two_sample_welch <- list(
  t_stat = as.numeric(t_welch$statistic),
  df = as.numeric(t_welch$parameter),
  p_value = t_welch$p.value,
  ci_lower = t_welch$conf.int[1],
  ci_upper = t_welch$conf.int[2],
  mean_diff = mean(setosa_sl) - mean(versi_sl),
  cohens_d = as.numeric(cohens_d_two)
)
t_student <- t.test(setosa_sl, versi_sl, var.equal=TRUE)
output$ttest_two_sample_student <- list(
  t_stat = as.numeric(t_student$statistic),
  df = as.numeric(t_student$parameter),
  p_value = t_student$p.value,
  ci_lower = t_student$conf.int[1],
  ci_upper = t_student$conf.int[2]
)

# 5. Paired t-test
cat("5. Paired t-test\n")
before <- c(85, 88, 90, 82, 87, 91, 86, 89, 84, 88)
after <- c(88, 91, 93, 86, 90, 95, 89, 92, 87, 91)
t_paired <- t.test(before, after, paired=TRUE)
cohens_d_paired <- effectsize::cohens_d(before, after, paired=TRUE)
output$ttest_paired <- list(
  t_stat = as.numeric(t_paired$statistic),
  df = as.numeric(t_paired$parameter),
  p_value = t_paired$p.value,
  ci_lower = t_paired$conf.int[1],
  ci_upper = t_paired$conf.int[2],
  mean_diff = mean(after - before),
  sd_diff = sd(after - before),
  cohens_d = as.numeric(cohens_d_paired)
)

# 6. One-way ANOVA
cat("6. One-way ANOVA\n")
aov_result <- aov(Sepal.Length ~ Species, data=iris)
aov_summary <- summary(aov_result)
ss_between <- aov_summary[[1]]["Species", "Sum Sq"]
ss_within <- aov_summary[[1]]["Residuals", "Sum Sq"]
df_between <- aov_summary[[1]]["Species", "Df"]
df_within <- aov_summary[[1]]["Residuals", "Df"]
ms_between <- ss_between / df_between
ms_within <- ss_within / df_within
eta_sq <- ss_between / (ss_between + ss_within)
omega_sq <- (df_between * (ms_between - ms_within)) / (ss_between + ss_within + ms_within)
output$anova_one_way <- list(
  f_stat = aov_summary[[1]]["Species", "F value"],
  df_between = df_between,
  df_within = df_within,
  ss_between = ss_between,
  ss_within = ss_within,
  ms_between = ms_between,
  ms_within = ms_within,
  p_value = aov_summary[[1]]["Species", "Pr(>F)"],
  eta_squared = eta_sq,
  omega_squared = omega_sq
)

# 7. Two-way ANOVA
cat("7. Two-way ANOVA\n")
set.seed(123)
n_cell <- 20
fa <- rep(c("A1", "A2"), each = n_cell * 2)
fb <- rep(rep(c("B1", "B2"), each = n_cell), 2)
y2 <- 50 + 3*(fa=="A2") + 2*(fb=="B2") + 1.5*(fa=="A2" & fb=="B2") + rnorm(n_cell*4, 0, 3)
df2 <- data.frame(A=fa, B=fb, Y=y2)
m2 <- lm(Y ~ A*B, data=df2)
a2 <- Anova(m2, type="II")
output$anova_two_way <- list(
  ss_A = a2["A", "Sum Sq"], ss_B = a2["B", "Sum Sq"],
  ss_AB = a2["A:B", "Sum Sq"], ss_error = a2["Residuals", "Sum Sq"],
  f_A = a2["A", "F value"], f_B = a2["B", "F value"], f_AB = a2["A:B", "F value"],
  p_A = a2["A", "Pr(>F)"], p_B = a2["B", "Pr(>F)"], p_AB = a2["A:B", "Pr(>F)"]
)

# 8. Linear Regression
cat("8. Linear Regression\n")
data(anscombe)
lm1 <- lm(y1 ~ x1, data=anscombe)
s1 <- summary(lm1)
output$regression_linear <- list(
  slope = coef(lm1)["x1"], intercept = coef(lm1)["(Intercept)"],
  r_squared = s1$r.squared, adj_r_squared = s1$adj.r.squared,
  residual_se = s1$sigma, f_stat = s1$fstatistic["value"],
  p_value = pf(s1$fstatistic["value"], s1$fstatistic["numdf"], s1$fstatistic["dendf"], lower.tail=FALSE),
  n = nobs(lm1)
)

# 9. Pearson Correlation - setosa SL vs versicolor SL (same as Python test)
cat("9. Pearson Correlation\n")
setosa_sl <- iris$Sepal.Length[iris$Species == "setosa"]
versi_sl <- iris$Sepal.Length[iris$Species == "versicolor"]
ct <- cor.test(setosa_sl, versi_sl)
output$correlation_pearson <- list(
  r_value = as.numeric(cor(setosa_sl, versi_sl)),
  t_stat = as.numeric(ct$statistic),
  df = as.numeric(ct$parameter),
  p_value = ct$p.value,
  ci_lower = ct$conf.int[1], ci_upper = ct$conf.int[2]
)

# 10. Spearman Correlation
cat("10. Spearman Correlation\n")
sp <- cor.test(setosa_sl, versi_sl, method="spearman")
output$correlation_spearman <- list(
  rho = as.numeric(sp$estimate), s_stat = as.numeric(sp$statistic), p_value = sp$p.value
)

# 11. Chi-square Test
cat("11. Chi-square Test\n")
cont <- matrix(c(50,30,20,60), nrow=2, byrow=TRUE)
chi2 <- chisq.test(cont)
cv <- sqrt(chi2$statistic / (sum(cont) * (min(dim(cont))-1)))
output$chi_square <- list(
  chi2_stat = as.numeric(chi2$statistic), df = as.numeric(chi2$parameter),
  p_value = chi2$p.value, cramers_v = as.numeric(cv),
  observed = as.vector(t(cont)), expected = as.vector(t(chi2$expected))
)

# 12. Mann-Whitney U
cat("12. Mann-Whitney U\n")
g1 <- c(23,25,28,30,32,35,27,29)
g2 <- c(31,33,36,38,40,42,35,37)
mw <- wilcox.test(g1, g2)
output$mann_whitney <- list(w_stat=as.numeric(mw$statistic), p_value=mw$p.value, n1=length(g1), n2=length(g2))

# 13. Kruskal-Wallis
cat("13. Kruskal-Wallis\n")
kw <- kruskal.test(list(c(7,14,14,13,12,9,6,14,12,7), c(15,17,13,15,15,13,9,12,10,6), c(6,7,11,8,7,6,10,5,4,9)))
output$kruskal_wallis <- list(chi2_stat=as.numeric(kw$statistic), df=as.numeric(kw$parameter), p_value=kw$p.value)

# 14. Wilcoxon Signed-Rank
cat("14. Wilcoxon Signed-Rank\n")
wsr <- wilcox.test(c(1.83,0.50,1.62,2.48,1.68,1.88,1.55,3.06,1.30), c(0.878,0.647,0.598,2.05,1.06,1.29,1.06,3.14,1.29), paired=TRUE)
output$wilcoxon_signed_rank <- list(v_stat=as.numeric(wsr$statistic), p_value=wsr$p.value)

# 15. Levene's Test
cat("15. Levene's Test\n")
lev <- leveneTest(Sepal.Length ~ Species, data=iris)
output$levene_test <- list(f_stat=lev["group","F value"], df1=lev["group","Df"], df2=lev["Residuals","Df"], p_value=lev["group","Pr(>F)"])

# 16. Bartlett's Test
cat("16. Bartlett's Test\n")
bart <- bartlett.test(Sepal.Length ~ Species, data=iris)
output$bartlett_test <- list(chi2_stat=as.numeric(bart$statistic), df=as.numeric(bart$parameter), p_value=bart$p.value)

# 17. Tukey HSD
cat("17. Tukey HSD\n")
tukey <- TukeyHSD(aov_result)
tdf <- as.data.frame(tukey$Species)
tukey_pairs <- list()
for (nm in rownames(tdf)) {
  tukey_pairs[[nm]] <- list(diff=tdf[nm,"diff"], lwr=tdf[nm,"lwr"], upr=tdf[nm,"upr"], p_adj=tdf[nm,"p adj"])
}
output$tukey_hsd <- tukey_pairs

# 18. Bonferroni correction
cat("18. Bonferroni\n")
pairwise_t <- pairwise.t.test(iris$Sepal.Length, iris$Species, p.adjust.method="bonferroni")
bonf_p <- list()
for (i in 1:nrow(pairwise_t$p.value)) {
  for (j in 1:ncol(pairwise_t$p.value)) {
    nm <- paste(rownames(pairwise_t$p.value)[i], colnames(pairwise_t$p.value)[j], sep="-")
    bonf_p[[nm]] <- pairwise_t$p.value[i,j]
  }
}
output$bonferroni <- bonf_p

# 19. Friedman Test
cat("19. Friedman\n")
friedman_data <- data.frame(
  y = c(7,3,6,5,6,4,5,3,4,3,8,5,7,5,6,4,5,4,3,2,9,6,8,5,7,5,6,4,5,3),
  group = factor(rep(c("A","B","C"), 10)),
  block = factor(rep(1:10, each=3))
)
fr <- friedman.test(y ~ group | block, data=friedman_data)
output$friedman <- list(chi2_stat=as.numeric(fr$statistic), df=as.numeric(fr$parameter), p_value=fr$p.value)

# 20. Nonlinear Regression (exponential)
cat("20. Nonlinear Regression\n")
x_nl <- 1:20
y_nl <- 2.5 * exp(0.3 * x_nl) + rnorm(20, 0, 2)
nl_fit <- nls(y_nl ~ a * exp(b * x_nl), start=list(a=2, b=0.3))
nl_summ <- summary(nl_fit)
output$regression_nonlinear <- list(
  a = coef(nl_fit)["a"], b = coef(nl_fit)["b"],
  residual_se = nl_summ$sigma,
  convergence = nl_fit$convInfo$isConv
)

# 21. Anderson-Darling Test
cat("21. Anderson-Darling\n")
ad <- ad.test(setosa_sl)
output$anderson_darling <- list(statistic=as.numeric(ad$statistic), p_value=ad$p.value)

# 22. Process Capability (manual reference)
cat("22. Process Capability\n")
cap_data <- c(10.1, 9.8, 10.2, 9.9, 10.0, 10.1, 9.7, 10.3, 10.0, 9.9,
              10.2, 10.1, 9.8, 10.0, 10.1, 9.9, 10.2, 10.0, 10.1, 9.8)
cap_mean <- mean(cap_data)
cap_sd <- sd(cap_data)  # overall std (ddof=1)
usl <- 11.0
lsl <- 9.0

# Moving range method for within-subgroup sigma
mr <- abs(diff(cap_data))
mr_bar <- mean(mr)
d2 <- 1.128  # for n=2 (individual observations)
sigma_within <- mr_bar / d2

# Pp/Ppk use overall std (same as our tool)
pp <- (usl - lsl) / (6 * cap_sd)
ppk <- min((usl - cap_mean)/(3*cap_sd), (cap_mean - lsl)/(3*cap_sd))

# Cp/Cpk use within-subgroup std (moving range)
cp <- (usl - lsl) / (6 * sigma_within)
cpk <- min((usl - cap_mean)/(3*sigma_within), (cap_mean - lsl)/(3*sigma_within))

output$process_capability <- list(
  mean=cap_mean, sd_overall=cap_sd, sigma_within=sigma_within,
  usl=usl, lsl=lsl,
  cp=cp, cpk=cpk, pp=pp, ppk=ppk
)

# Write JSON
cat("Writing reference_values.json...\n")
json_out <- toJSON(output, auto_unbox=TRUE, digits=15, pretty=TRUE)
writeLines(json_out, "D:/learn/claudecode/stats-cli/tests/cross_validation/reference_values.json")
cat("Done!\n")