library(lme4)      # for mixed-effects models via lmer()
library(lmerTest)
library(emmeans)
emm_options(
  lmer.df = "satterthwaite",
  lmerTest.limit = 80000
)

path <- "exp_data/analysis/cv_model_performance_spearman_corr.csv"
corrs <- read.csv(path)

# Fisher z with clipping
corrs$z <- atanh(pmax(pmin(corrs$cv_corr, 1 - 1e-6), -1 + 1e-6))
corrs$generator <- factor(corrs$generator)
# rename condition to sampling
corrs$sampling <- factor(corrs$condition)

corrs$seed <- factor(corrs$seed)
corrs$model <- factor(corrs$model)
corrs$subj_id <- factor(corrs$subj_id)
corrs$i_trial <- factor(corrs$i_trial)

# create globally-unique grouping IDs to avoid any ambiguity
corrs$seed_id <- interaction(corrs$generator, corrs$sampling, corrs$seed, drop = TRUE)
corrs$subj_uid <- interaction(corrs$seed_id, corrs$subj_id, drop = TRUE)
corrs$trial_uid<- interaction(corrs$seed_id, corrs$i_trial, drop = TRUE)

res <- data.frame(
  cond = character(0),
  contrast = character(0),
  estimate = numeric(0),
  SE = numeric(0),
  df = numeric(0),
  t.ratio = numeric(0),
  p.value   = numeric(0),
  p.value.fdr = numeric(0),
  stringsAsFactors = FALSE
)

fit <- lmer(
  z ~ model * generator * sampling +
    (1 | seed_id) +
    (1 | subj_uid) +
    (1 | trial_uid),
  data = corrs,
  REML = TRUE
)

cat("fit summary")
print(summary(fit))

cat("check convergence")
print(fit@optinfo$conv$lme4$messages)
print(fit@optinfo$conv$opt)

cat("check singularity")
print(isSingular(fit, tol = 1e-4))
cat("\nVariance components:\n")
print(VarCorr(fit), comp = c("Variance", "Std.Dev."))

cat("parameter count\n")
p_fixed <- length(fixef(fit))
p_theta <- length(getME(fit, "theta"))
p_sigma <- 1
p_total <- p_fixed + p_theta + p_sigma

cat(sprintf("Fixed effects (beta): %d\n", p_fixed))
cat(sprintf("Random intercept cov param: %d\n", p_theta))
cat(sprintf("Residual SD: %d\n", p_sigma))
cat(sprintf("Total (beta + theta + sigma): %d\n", p_total))

# ==============================================================================
# MODEL SELECTION: Compare Full Model vs. Reduced Variants
# ==============================================================================

# Note: We use REML = FALSE for model comparison (AIC/BIC validity)
# We use calc.derivs = FALSE to speed up fitting since we only need the logLik, not SEs here.

cat("\n==================== FITTING MODEL VARIANTS ====================\n")

# 1. The Full Model (3-way interaction)
message("Fitting: Full Model...")
m_full <- lmer(
  z ~ model * generator * sampling + 
    (1 | seed_id) + (1 | subj_uid) + (1 | trial_uid),
  data = corrs, REML = FALSE, control = lmerControl(calc.derivs = FALSE)
)

# 2. No Three-Way Interaction (Main effects + all 2-way interactions)
# Formula: (A+B+C)^2 expands to A+B+C + A:B + A:C + B:C
message("Fitting: No 3-Way Interaction...")
m_no_3way <- lmer(
  z ~ (model + generator + sampling)^2 + 
    (1 | seed_id) + (1 | subj_uid) + (1 | trial_uid),
  data = corrs, REML = FALSE, control = lmerControl(calc.derivs = FALSE)
)

# 3. No Model:Sampling Interaction
# Formula includes Model*Gen and Gen*Samp, but omits Model*Samp
message("Fitting: No Model:Sampling...")
m_no_mod_samp <- lmer(
  z ~ model * generator + generator * sampling + 
    (1 | seed_id) + (1 | subj_uid) + (1 | trial_uid),
  data = corrs, REML = FALSE, control = lmerControl(calc.derivs = FALSE)
)

# 4. No Model:Generator Interaction
# Formula includes Model*Samp and Samp*Gen, but omits Model*Gen
message("Fitting: No Model:Generator...")
m_no_mod_gen <- lmer(
  z ~ model * sampling + sampling * generator + 
    (1 | seed_id) + (1 | subj_uid) + (1 | trial_uid),
  data = corrs, REML = FALSE, control = lmerControl(calc.derivs = FALSE)
)

# 5. No Generator:Sampling Interaction
# Formula includes Model*Samp and Model*Gen, but omits Gen*Samp
message("Fitting: No Generator:Sampling...")
m_no_gen_samp <- lmer(
  z ~ model * sampling + model * generator + 
    (1 | seed_id) + (1 | subj_uid) + (1 | trial_uid),
  data = corrs, REML = FALSE, control = lmerControl(calc.derivs = FALSE)
)

# 6. No Model Interactions (Model is additive only)
# Model has a consistent effect regardless of Gen or Samp
message("Fitting: No Model Interactions...")
m_no_mod_int <- lmer(
  z ~ model + generator * sampling + 
    (1 | seed_id) + (1 | subj_uid) + (1 | trial_uid),
  data = corrs, REML = FALSE, control = lmerControl(calc.derivs = FALSE)
)

# ==============================================================================
# COMPARE AIC / BIC
# ==============================================================================

cat("\n==================== MODEL COMPARISON TABLE ====================\n")
# anova() with multiple objects produces the comparison table
comp_table <- anova(m_full, m_no_3way, m_no_mod_samp, m_no_mod_gen, m_no_gen_samp, m_no_mod_int)

# Sort by AIC (lowest is best)
comp_table_sorted <- comp_table[order(comp_table$AIC), ]
print(comp_table_sorted)

# Quick check: Is Full Model the winner?
winner <- rownames(comp_table_sorted)[1]
cat(sprintf("\n The best fitting model (lowest AIC) is: %s\n", winner))

# Overall model comparison pooled across generator × sampling
# the estimated mean for each model is averaged over the six generator×sampling cells with equal weight
emm_overall <- emmeans(fit, ~ model, weights = "equal") # marginal means averaged over generator x sampling
cmp_overall <- pairs(emm_overall, adjust = "fdr")
 
overall_df <- as.data.frame(summary(cmp_overall))
write.csv(
  overall_df,
  "exp_data/analysis/pooled_pairwise_satterthwaite.csv",
  row.names = FALSE
)
