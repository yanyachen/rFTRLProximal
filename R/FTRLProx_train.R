#' @title FTRL-Proximal Linear Model Training
#'
#' @description
#' An advanced interface for training FTRL-Proximal online learning model.
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
#' @param watchlist a named list, used to specify validation set monitoring during training.
#' @param eval a custimized evaluation function, the first argument shoule be prediction, the second argument shoule be label.
#' @param patience The number of rounds with no improvement in the evaluation metric in order to stop the training. User can specify 0 to disable early stopping.
#' @param maximize whether to maximize the evaluation metric.
#' @param nthread number of parallel threads used to run ftrl. Please set to 1 if your feature set is not sparse enough.
#' @param verbose logical value. Indicating if the progress bar is displayed or not.
#' @return a FTRL-Proximal linear model object
#' @references
#' H. B. McMahan, G. Holt, D. Sculley, et al. "Ad click prediction: a view from the trenches".
#' In: _The 19th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining,
#' KDD 2013, Chicago, IL, USA, August 11-14, 2013_. Ed. by I. S.Dhillon, Y. Koren, R. Ghani,
#' T. E. Senator, P. Bradley, R. Parekh, J. He, R. L. Grossman and R. Uthurusamy. ACM, 2013, pp. 1222-1230.
#' DOI: 10.1145/2487575.2488200. <URL: \url{http://doi.acm.org/10.1145/2487575.2488200}>.
#' @examples
#' library(data.table)
#' library(FeatureHashing)
#' library(MLmetrics)
#' data(ipinyou)
#' m.train <- FTRLProx_Hashing(~ 0 + ., ipinyou.train[, -"IsClick", with = FALSE],
#'                             hash.size = 2^13, signed.hash = FALSE, verbose = TRUE,
#'                             label = as.numeric(ipinyou.train$IsClick))
#' m.test <- FTRLProx_Hashing(~ 0 + ., ipinyou.test[,-"IsClick", with = FALSE],
#'                            hash.size = 2^13, signed.hash = FALSE, verbose = TRUE,
#'                            label = as.numeric(ipinyou.test$IsClick))
#' ftrl_model <- FTRLProx_train(data = m.train, model = NULL,
#'                              family = "binomial",
#'                              params = list(alpha = 0.01, beta = 0.1,
#'                                            l1 = 0.1, l2 = 0.1, dropout = 0), epoch = 50,
#'                              watchlist = list(test = m.test),
#'                              eval = AUC, patience = 5, maximize = TRUE,
#'                              nthread = 1, verbose = TRUE)
#' pred_ftrl <- FTRLProx_predict(ftrl_model, m.test)
#' AUC(pred_ftrl, as.numeric(ipinyou.test$IsClick))
#' @importFrom magrittr %>% %T>%
#' @export

FTRLProx_train <- function(data, model = NULL,
                           family = c("gaussian", "binomial", "poisson"),
                           params = list(alpha = 0.1, beta = 1.0, l1 = 1.0, l2 = 1.0, dropout = 0), epoch = 1,
                           watchlist = list(), eval = NULL, patience = 0, maximize = NULL, nthread = 1, verbose = TRUE) {
  # Model Initialization
  if (is.null(model)) {
    model_state_param <- list(state = NULL, family = family, params = params, mapping = data$Mapping)
  } else {
    stopifnot(identical(model$mapping, data$Mapping))
    model_state_param <- list(state = model$state, family = model$family, params = model$params, mapping = model$mapping)
  }
  # Model Computing
  watchlist_len <- length(watchlist)
  if (watchlist_len == 0) {
    FTRLProx_State <- FTRLProx_train_spMatrix(x = data$X, y = data$y,
                                              state = model_state_param$state,
                                              family = model_state_param$family, params = model_state_param$params,
                                              epoch = epoch, nthread = nthread, verbose = verbose)
  } else {
    FTRLProx_State <- model_state_param$state
    watchlist_dt <- lapply(X = seq_len(1 + watchlist_len), FUN = function(i) vector(mode = "double", length = epoch)) %>%
      data.table::as.data.table(.) %T>%
      data.table::setnames(., old = names(.), new = c("train", names(watchlist)))
    for (i in seq_len(epoch)) {
      FTRLProx_State <- FTRLProx_train_spMatrix(x = data$X, y = data$y,
                                                state = FTRLProx_State,
                                                family = model_state_param$family, params = model_state_param$params,
                                                epoch = 1, nthread = nthread, verbose = FALSE)
      for (j in seq_along(watchlist_dt)) {
        if (j == 1) {
          perf <- eval(predict_spMatrix(data$X, FTRLProx_State$w, model_state_param$family) %>% as.double(.),
                       data$y)
        } else {
          perf <- eval(predict_spMatrix(watchlist[[j - 1]]$X, FTRLProx_State$w, model_state_param$family) %>% as.double(.),
                       watchlist[[j - 1]]$y)
        }
        data.table::set(watchlist_dt,
                        i = as.integer(i),
                        j = as.integer(j),
                        value = perf)
      }
      if (isTRUE(verbose)) {
        Perf_Print(Round = i, Name = names(watchlist_dt), Value = as.double(watchlist_dt[i, ]))
      }
      if (patience != 0 & i > patience) {
        if (isTRUE(maximize)) {
          round_max <- which.max(watchlist_dt[[2]])
          if (round_max == i - patience) break;
        } else {
          round_min <- which.max(watchlist_dt[[2]])
          if (round_min == i - patience) break;
        }
      }
    }
    perf_dt <- watchlist_dt[seq_len(i), ]
  }
  # Model Object
  FTRLProx_Model <- list(state = FTRLProx_State,
                         family = model_state_param$family, params = model_state_param$params,
                         mapping = model_state_param$mapping)
  class(FTRLProx_Model) <- "ftrl.model"
  return(FTRLProx_Model)
}
