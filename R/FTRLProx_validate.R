#' @title FTRL-Proximal Linear Model Validation
#'
#' @description
#' An advanced interface for FTRL-Proximal online learning model validation.
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
#' @param val_x a transposed \code{dgCMatrix} for validation.
#' @param val_y a vector containing labels for validation.
#' @param eval a evaluation metrics computing function, the first argument shoule be prediction, the second argument shoule be label.
#' @param verbose logical value. Indicating if the validation result for each epoch is displayed or not.
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
#'                             hash.size = 2^13, signed.hash = FALSE, verbose = TRUE)
#' m.test <- FTRLProx_Hashing(~ 0 + ., ipinyou.test[,-"IsClick", with = FALSE],
#'                            hash.size = 2^13, signed.hash = FALSE, verbose = TRUE)
#' ftrl_model_val <- FTRLProx_validate(x = m.train, y = as.numeric(ipinyou.train$IsClick),
#'                                     family = "binomial",
#'                                     params = list(alpha = 0.01, beta = 0.1, l1 = 1.0, l2 = 1.0),
#'                                     epoch = 20,
#'                                     val_x = m.test,
#'                                     val_y = as.numeric(ipinyou.test$IsClick),
#'                                     eval = AUC, verbose = TRUE)
#' @export

FTRLProx_validate <- function(x, y, family = c("gaussian", "binomial", "poisson"),
                              params = list(alpha = 0.1, beta = 1.0, l1 = 1.0, l2 = 1.0), epoch = 1,
                              val_x, val_y, eval, verbose = TRUE) {
  # Feature Mapping
  mapping <- FeatureHashing::hash.mapping(x)
  # Model Computing
  weight_perf <- FTRLProx_validate_spMatrix(x = x, y = y, family = family, params = params, epoch = epoch,
                                            val_x = val_x, val_y = val_y, eval = eval, verbose = verbose)
  weight_mat <- NULL
  # Model Object
  FTRLProx <- list(weight = as.numeric(weight_perf$weight), weight_mat = weight_mat, mapping = mapping,
                   family = family, params = params, epoch = epoch,
                   eval_train = as.numeric(weight_perf$eval_train), 
                   eval_val = as.numeric(weight_perf$eval_val))
  return(FTRLProx)
}

