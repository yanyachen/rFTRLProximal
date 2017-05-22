#' @title FTRL-Proximal Linear Model Cross Validation
#'
#' @description
#' An advanced interface for FTRL-Proximal online learning model cross validation.
#'
#' @param data a object of class \code{ftrl.Dataset}.
#' @param model a previously built model object to continue the training from.
#' @param family link function to be used in the model. "gaussian", "binomial" and "poisson" are avaliable.
#' @param params a list of parameters of FTRL-Proximal Algorithm.
#' \itemize{
#'   \item \code{alpha} alpha in the per-coordinate learning rate
#'   \item \code{beta} beta in the per-coordinate learning rate
#'   \item \code{l1} L1 regularization parameter
#'   \item \code{l2} L2 regularization parameter
#'   \item \code{dropout} percentage of the input features to drop from each sample
#' }
#' @param epoch The number of iterations over training data to train the model.
#' @param folds \code{list} provides a possibility of using a list of pre-defined CV folds (each element must be a vector of fold's indices).
#' @param eval a custimized evaluation function, the first argument shoule be prediction, the second argument shoule be label.
#' @param patience The number of rounds with no improvement in the evaluation metric in order to stop the training. User can specify 0 to disable early stopping.
#' @param maximize whether to maximize the evaluation metric.
#' @param nthread number of parallel threads used to run ftrl. Please set to 1 if your feature set is not sparse enough.
#' @param verbose logical value. Indicating if the progress bar is displayed or not.
#' @return a list with the following elements is returned:
#' \itemize{
#'   \item \code{dt} a data.table with each mean and standard deviation stat for training set and test set
#'   \item \code{pred} a numerical vector with predictions for each CV-fold for the model having been trained on the data in all other folds.
#' }
#' @references
#' H. B. McMahan, G. Holt, D. Sculley, et al. "Ad click prediction: a view from the trenches".
#' In: _The 19th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining,
#' KDD 2013, Chicago, IL, USA, August 11-14, 2013_. Ed. by I. S.Dhillon, Y. Koren, R. Ghani,
#' T. E. Senator, P. Bradley, R. Parekh, J. He, R. L. Grossman and R. Uthurusamy. ACM, 2013, pp. 1222-1230.
#' DOI: 10.1145/2487575.2488200. <URL: \url{http://doi.acm.org/10.1145/2487575.2488200}>.
#' @examples
#' library(FeatureHashing)
#' library(data.table)
#' library(rBayesianOptimization)
#' library(MLmetrics)
#' data(ipinyou)
#' m.train <- FTRLProx_Hashing(~ 0 + ., ipinyou.train[, -"IsClick", with = FALSE],
#'                             hash.size = 2^13, signed.hash = FALSE, verbose = TRUE,
#'                             label = as.numeric(ipinyou.train$IsClick))
#' ftrl_model_cv <- FTRLProx_cv(data = m.train, model = NULL,
#'                              family = "binomial",
#'                              params = list(alpha = 0.01, beta = 0.1,
#'                                            l1 = 1.0, l2 = 1.0, dropout = 0), epoch = 50,
#'                              folds = KFold(as.numeric(ipinyou.train$IsClick), nfolds = 5,
#'                                            stratified = FALSE, seed = 0),
#'                              eval = AUC, patience = 5, maximize = TRUE,
#'                              nthread = 1, verbose = TRUE)
#' @importFrom magrittr %>% %T>%
#' @importFrom foreach %do%
#' @importFrom stats sd
#' @export

FTRLProx_cv <- function(data, model = NULL,
                        family = c("gaussian", "binomial", "poisson"),
                        params = list(alpha = 0.1, beta = 1.0, l1 = 1.0, l2 = 1.0), epoch = 1,
                        folds, eval = NULL, patience = 0, maximize = NULL, nthread = 1, verbose = TRUE) {
  # Model Initialization
  if (is.null(model)) {
    model_state_param <- list(state = NULL, family = family, params = params, mapping = data$Mapping)
  } else {
    stopifnot(identical(model$mapping, data$Mapping))
    model_state_param <- list(state = model$state, family = model$family, params = model$params, mapping = model$mapping)
  }
  # CV Computing
  FTRLProx_State_List <- lapply(X = seq_along(folds), FUN = function(i) model_state_param$state)
  train_data_list <- lapply(X = seq_along(folds), FUN = function(i) slice(data, -folds[[i]]))
  test_data_list <- lapply(X = seq_along(folds), FUN = function(i) slice(data, folds[[i]]))
  Perf_CV <- lapply(X = 1:4, function(i) vector(mode = "double", length = epoch)) %>%
    data.table::as.data.table(.) %T>%
    data.table::setnames(., old = names(.), new = c("train_mean", "train_sd", "test_mean", "test_sd"))
  for (i in seq_len(epoch)) {
    for (j in seq_along(folds)) {
      FTRLProx_State_List[[j]] <- FTRLProx_train_spMatrix(x = train_data_list[[j]]$X, y = train_data_list[[j]]$y,
                                                          state = FTRLProx_State_List[[j]],
                                                          family = model_state_param$family, params = model_state_param$params,
                                                          epoch = 1, nthread = nthread, verbose = FALSE)
    }
    train_perf_vec <- foreach::foreach(j = seq_along(folds), .combine = "c") %do% {
      eval(predict_spMatrix(train_data_list[[j]]$X, FTRLProx_State_List[[j]]$w, model_state_param$family),
           train_data_list[[j]]$y)
    }
    test_perf_vec <- foreach::foreach(j = seq_along(folds), .combine = "c") %do% {
      eval(predict_spMatrix(test_data_list[[j]]$X, FTRLProx_State_List[[j]]$w, model_state_param$family),
           test_data_list[[j]]$y)
    }
    data.table::set(Perf_CV,
                    i = as.integer(i),
                    j = 1L:4L,
                    value = list(mean(train_perf_vec), sd(train_perf_vec),
                                 mean(test_perf_vec), sd(test_perf_vec)))
    if (isTRUE(verbose)) {
      Perf_Print(Round = i, Name = names(Perf_CV), Value = as.double(Perf_CV[i, ]))
    }
    if (patience != 0 & i > patience) {
      if (isTRUE(maximize)) {
        round_max <- which.max(Perf_CV[[3]])
        if (round_max == i - patience) break;
      } else {
        round_min <- which.min(Perf_CV[[3]])
        if (round_min == i - patience) break;
      }
    }
  }
  Pred_CV <- foreach::foreach(j = seq_along(folds), .combine = "c") %do% {
    predict_spMatrix(test_data_list[[j]]$X, FTRLProx_State_List[[j]]$w, model_state_param$family)
  } %>% magrittr::extract(., order(unlist(folds)))
  # Return Performance and OOB Prediction
  list(dt = Perf_CV[seq_len(i), ],
       pred = Pred_CV)
}
