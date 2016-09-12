#' @title FTRL-Proximal Linear Model Training
#'
#' @description
#' An advanced interface for training FTRL-Proximal online learning model.
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
#' @param bagging_seeds a vector containing random seeds for shuffling data.
#'   If provided, use parallel foreach to fit each model. Must register parallel before hand, such as doParallel or others.
#' @param verbose logical value. Indicating if the progress bar is displayed or not.
#' @return a FTRL-Proximal linear model object
#' @examples
#' library(data.table)
#' library(FeatureHashing)
#' data(ipinyou)
#' m.train <- FTRLProx_Hashing(~ 0 + ., ipinyou.train[, -"IsClick", with = FALSE],
#'                             hash.size = 2^13, signed.hash = FALSE, verbose = TRUE)
#' m.test <- FTRLProx_Hashing(~ 0 + ., ipinyou.test[,-"IsClick", with = FALSE],
#'                            hash.size = 2^13, signed.hash = FALSE, verbose = TRUE)
#' ftrl_model <- FTRLProx_train(m.train, y = as.numeric(ipinyou.train$IsClick),
#'                              family = "binomial",
#'                              params = list(alpha = 0.01, beta = 0.1, l1 = 1.0, l2 = 1.0),
#'                              epoch = 10, verbose = TRUE)
#' ftrl_model_bagging <- FTRLProx_train(m.train, y = as.numeric(ipinyou.train$IsClick),
#'                                      family = "binomial",
#'                                      params = list(alpha = 0.01, beta = 0.1, l1 = 1.0, l2 = 1.0),
#'                                      epoch = 10, bagging_seeds = 1:10, verbose = FALSE)
#' pred_ftrl <- FTRLProx_predict(ftrl_model, newx = m.test)
#' pred_ftrl_bagging <- FTRLProx_predict(ftrl_model_bagging, newx = m.test)
#' MLmetrics::AUC(pred_ftrl, as.numeric(ipinyou.test$IsClick))
#' MLmetrics::AUC(pred_ftrl_bagging, as.numeric(ipinyou.test$IsClick))
#' @importFrom foreach %dopar%
#' @export

FTRLProx_train <- function(x, y, family = c("gaussian", "binomial", "poisson"),
                           params = list(alpha = 0.1, beta = 1.0, l1 = 1.0, l2 = 1.0), epoch = 1,
                           bagging_seeds = NULL, verbose = TRUE) {
  # Feature Mapping
  mapping <- FeatureHashing::hash.mapping(x)
  # Model Computing
  if (is.null(bagging_seeds) == TRUE) {
    weight <- FTRLProx_train_spMatrix(x = x, y = y, family = family, params = params, epoch = epoch, verbose = verbose)
    weight_mat <- NULL
  } else {
    weight_mat <- foreach::foreach(i = bagging_seeds, .combine = "cbind") %dopar% {
      set.seed(i)
      shuffle_i <- sample(1:length(y), size = length(y), replace = FALSE)
      weight <- FTRLProx_train_spMatrix(x = x[,shuffle_i], y = y[shuffle_i],
                                        family = family, params = params, epoch = epoch, verbose = FALSE)
    }
    weight <- Matrix::rowMeans(as.matrix(weight_mat))
  }
  # Model Object
  FTRLProx <- list(weight = weight, weight_mat = weight_mat, mapping = mapping,
                   family = family, params = params, epoch = epoch,
                   bagging_seeds = bagging_seeds, verbose = verbose)
  return(FTRLProx)
}
utils::globalVariables(c("i"))
