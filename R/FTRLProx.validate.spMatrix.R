#' @title (Depreciated) FTRL-Proximal Linear Model Validation Function
#'
#' @description
#' FTRLProx_validate_spMatrix validates the performance of FTRL-Proximal online learning model.
#' This function is an Pure R implementation.
#' This function is used internally and is not intended for end-user direct usage.
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
#' @export

FTRLProx.validate.spMatrix <- function(x, y, family, params, epoch, val_x, val_y, eval, verbose) {
  # Family
  family <- match.arg(arg = family, choices = c("gaussian", "binomial", "poisson"))
  PredTransform <- switch(family,
                          gaussian = function(x) x,
                          binomial = function(x) 1 / (1 + exp(-x)),
                          poisson = function (x) exp(x))
  FirstOrderGradient <- switch(family,
                               gaussian = function(p, y) p - y,
                               binomial = function(p, y) p - y,
                               poisson = function(p, y) p - y)
  # Hyperparameter
  alpha <- params$alpha
  beta <- params$beta
  l1 <- params$l1
  l2 <- params$l2
  # Model Initialization
  z <- numeric(nrow(x))
  n <- numeric(nrow(x))
  w <- numeric(nrow(x))
  # Model Prediction
  p <- numeric(length(y))
  # Training and Validation Performance
  eval_train <- numeric(epoch)
  eval_val <- numeric(epoch)
  # Computing
  ## Weight Update Function
  weight_update <- function(alpha, beta, l1, l2, z, n) {
    (-(z - sign(z) * l1) / ((beta + sqrt(n)) / alpha + l2)) * as.numeric(abs(z) > l1)
  }
  ## Non-Zero Feature Count for in spMatrix
  non_zero_count <- diff.default(x@p)
  for (r in 1:epoch) {
    for (t in seq_len(ncol(x))) {
      ## Non-Zero Feature Index in spMatrix
      non_zero_index <- x@p[t] + seq_len(non_zero_count[t])
      ## Non-Zero Feature Index for each sample
      i <- x@i[non_zero_index] + 1
      ## Non-Zero Feature Value for each sample
      x_i <- x@x[non_zero_index]
      ## Computing Weight and Prediction
      w[i] <- weight_update(alpha, beta, l1, l2, z[i], n[i])
      p[t] <- PredTransform(sum(x_i * w[i]))
      ## Updating Model
      g_i <- FirstOrderGradient(p[t], y[t]) * x_i
      s_i <- (sqrt(n[i] + g_i^2) - sqrt(n[i])) / alpha
      z[i] <- z[i] + g_i - s_i * w[i]
      n[i] <- n[i] + g_i^2
    }
    eval_train[r] <- eval(p, y)
    eval_val[r] <- eval(FTRLProx.predict.spMatrix(val_x, w, family), val_y)
    if (verbose == TRUE) {
      paste("[", r, "]", " \t train: ", eval_train[r], " \t validation: ", eval_val[r], "\n", sep = "") %>% cat(.)
    }
  }
  # Retrun FTRL Proximal Model Weight and Performance
  return(list(weight = w,
              eval_train = eval_train,
              eval_val = eval_val))
}
