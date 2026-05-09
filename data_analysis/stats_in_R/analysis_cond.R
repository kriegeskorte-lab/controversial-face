library(lme4)      # for mixed-effects models via lmer()
library(lmerTest)
library(emmeans)
library(brms)
library(RePsychLing)

path <- "exp_data/analysis/cv_model_performance_spearman_corr.csv"
corrs <- read.csv(path)
corrs$z <-atanh(pmax(pmin(corrs$cv_corr, 1 - 1e-6), -1 + 1e-6))
gens  <- unique(corrs$generator)
conds <- unique(corrs$condition)

corrs$generator <- as.factor(corrs$generator)
corrs$condition <- as.factor(corrs$condition)
corrs$seed      <- as.factor(corrs$seed)
corrs$model     <- as.factor(corrs$model)
corrs$subj_id   <- as.factor(corrs$subj_id)
corrs$i_trial   <- as.factor(corrs$i_trial)

res <- data.frame(
  generator = character(0),
  condition = character(0),
  contrast  = character(0),
  estimate  = numeric(0),
  SE        = numeric(0),
  df        = numeric(0),
  t.ratio   = numeric(0),
  p.value   = numeric(0),
  p.value.fdr = numeric(0),
  stringsAsFactors = FALSE
)

emm_options(
  lmer.df = "satterthwaite",
  lmerTest.limit = 13000
)

anova_results <- list()
warnings_list <- list()  # Store warnings

for (gen in gens) {
  for (cond in conds) {
    cat("Generator:", gen, "Condition:", cond, "\n")
    corr_sub <- subset(corrs, generator == gen & condition == cond)

    # a parsimonious random-intercepts model
    # showed stable convergence while capturing the key sources of variance in the experimental design.
    model_formula <- as.formula(z ~ model + (1 | seed:subj_id) + (1 | seed:i_trial))

    # We considered the following more complex random-effects structures, including random intercepts 
    # for seed and random slopes of model, but these yielded convergence failures or 
    # unstable parameter estimates across different numerical optimizers. 
    # model_formula1 <- as.formula(z ~ model + (1 + model || seed:subj_id) + (1 + model || seed:i_trial))
    # model_formula2 <- as.formula(z ~ model + (1 | seed:subj_id) + (1 + model || seed:i_trial))
    # model_formula3 <- as.formula(z ~ model + (1 + model || seed:subj_id) + (1 | seed:i_trial))
    
    fit <- lmer(model_formula, data = corr_sub, REML = TRUE)
    
    # fit1 <- lmer(model_formula1, data = corr_sub, REML = TRUE)
    # fit3 <- lmer(model_formula3, data = corr_sub, REML = TRUE)
    # fit2 <- lmer(model_formula2, data = corr_sub, REML = TRUE)
    # print("lmer default optimizer:")
    # print(summary(fit2))
    # print("running refit:")
    # refit <- allFit(fit2)
    # print(summary(refit))

    emm  <- emmeans(fit, specs = ~ model)
    cmp  <- pairs(emm, adjust = "fdr")
    df_c <- as.data.frame(summary(cmp))
    df_c$p.value.fdr <- df_c$p.value
    df_c$p.value <- NULL
    
    df_c$generator <- gen
    df_c$condition <- cond
    df_c <- df_c[, c("generator", "condition", "contrast",
                     "estimate", "SE", "t.ratio", "p.value.fdr")]
    # Append to the main results dataframe
    res <- rbind(res, df_c)
    
  }
}

save_path <- "exp_data/analysis/cond_pairwise_satterthwaite.csv"
write.csv(res, save_path, row.names=FALSE)