library(tidyverse)
library(pROC)

# read in prediction files
df <- readr::read_csv('~/projects/TwitterR01/AnnotationProjects/classification/results/LogisticRegression_pro_vape_preds.csv')
df2 <- readr::read_csv('~/projects/TwitterR01/AnnotationProjects/classification/results/lstm_comvape_glove_preds.csv') %>%
  rename(y_pred2 = y_pred)
# merge prediction files
df <- df %>%
  left_join(df2, on = 'X1')
# create AUC objects w/ Logistic Regression as baseline
lr <- pROC::auc(df$y_true, df$y_pred)
m2 <- pROC::auc(df$y_true, df$y_pred2)

# run DeLong test comparing the two curves
pROC::roc.test(lr, m2)
