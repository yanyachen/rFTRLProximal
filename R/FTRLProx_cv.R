#' @title FTRL-Proximal Linear Model Cross Validation
#'
#' @description
#' An advanced interface for FTRL-Proximal online learning model cross validation.
#'
#' @param x a transposed \code{dgCMatrix}.
#' @param y a vector containing labels.
#' @param family link function to be used in the model. "gaussian", "binomial" and "poisson" are avaliable.
#' @param params a list of parameters of FTRL-Proximal Algorithm.
#' \itemize{
#'   \item \code{alpha} alpha in the per-coordinate learning rate
#'   \item \code{beta} beta in the per-coordinate learning rate
#'   \item \code{l1} L1 regularization parameter
#'   \item \code{l2} L2 regularization parameter
#' }
#' @param epoch The number of iterations over training data to train the model.
#' @param folds \code{list} provides a possibility of using a list of pre-defined CV folds (each element must be a vector of fold's indices).
#' @param eval a evaluation metrics computing function, the first argument shoule be prediction, the second argument shoule be label.
#' @return a list with the following elements is returned:
#' \itemize{
#'   \item \code{dt} a data.table with each mean and standard deviation stat for training set and test set
#'   \item \code{pred} a numerical vector with predictions for each CV-fold for the model having been trained on the data in all other folds.
#' }
#' @examples
#' library(FeatureHashing)
#' library(data.table)
#' library(rBayesianOptimization)
#' data(ipinyou)
#' m.train <- FTRLProx_Hashing(~ 0 + ., ipinyou.train[,-"IsClick", with = FALSE],
#'                             hash.size = 2^13, signed.hash = FALSE, verbose = TRUE)
#' ftrl_model_cv <- FTRLProx_cv(x = m.train, y = as.numeric(ipinyou.train$IsClick),
#'                              family = "binomial",
#'                              params = list(alpha = 0.01, beta = 0.1, l1 = 1.0, l2 = 1.0),
#'                              epoch = 10,
#'                              folds = KFold(as.numeric(ipinyou.train$IsClick), nfolds = 5),
#'                              eval = MLmetrics::AUC)
#' @importFrom magrittr %>%
#' @importFrom foreach %do%
#' @importFrom stats sd
#' @export

FTRLProx_cv <- function(x, y, family = c("gaussian", "binomial", "poisson"),
                        params = list(alpha = 0.1, beta = 1.0, l1 = 1.0, l2 = 1.0), epoch = 1,
                        folds, eval) {
  Perf_Pred_List <- foreach::foreach(i = seq_along(folds)) %do% {
    FTRLProx <- FTRLProx_validate(x = slice(x, -folds[[i]]), y = y[-folds[[i]]], family = family,
                                  params = params, epoch = epoch,
                                  val_x = slice(x, folds[[i]]), val_y = y[folds[[i]]], eval = eval,
                                  verbose = FALSE)
    Pred <- FTRLProx_predict(FTRLProx, newx = slice(x, folds[[i]]))
    Perf_Train <- FTRLProx$eval_train
    Perf_Val <- FTRLProx$eval_val
    list(Pred = Pred, Perf_Train = Perf_Train, Perf_Val = Perf_Val)
  }
  ID_CV <- foreach::foreach(i = seq_along(folds), .combine = "c") %do% {
    folds[[i]]
  }
  Pred_CV <- foreach::foreach(i = seq_along(Perf_Pred_List), .combine = "c") %do% {
    Perf_Pred_List[[i]]$Pred
  } %>% magrittr::extract(., order(ID_CV))
  Perf_Train_CV <- foreach::foreach(i = seq_along(Perf_Pred_List), .combine = "cbind") %do% {
    Perf_Pred_List[[i]]$Perf_Train
  }
  Perf_Val_CV <- foreach::foreach(i = seq_along(Perf_Pred_List), .combine = "cbind") %do% {
    Perf_Pred_List[[i]]$Perf_Val
  }
  Perf_CV = data.table::data.table(Train_Mean = apply(Perf_Train_CV, 1, mean),
                                   Train_SD = apply(Perf_Train_CV, 1, sd),
                                   Validation_Mean = apply(Perf_Val_CV, 1, mean),
                                   Validation_SD = apply(Perf_Val_CV, 1, sd))
  list(dt = Perf_CV,
       pred = Pred_CV)
}
utils::globalVariables(c("i", "."))
