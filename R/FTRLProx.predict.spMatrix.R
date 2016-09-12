#' @title (Depreciated) FTRL-Proximal Linear Model Predicting Function
#'
#' @description
#' FTRLProx.predict.spMatrix predicts values based on linear model weights.
#' This function is an Pure R implementation.
#' This function is used internally and is not intended for end-user direct usage.
#'
#' @param x a transposed \code{dgCMatrix} object.
#' @param w an vector of linear model weights.
#' @param family link function to be used in the model. "gaussian", "binomial" and "poisson" are avaliable.
#' @return an vector of linear model predicted values
#' @keywords internal
#' @export

FTRLProx.predict.spMatrix <- function(x, w, family) {
  # Family
  PredTransform <- switch(family,
                          gaussian = function(x) x,
                          binomial = function(x) 1 / (1 + exp(-x)),
                          poisson = function (x) exp(x))
  # Prediction
  PredTransform(as.numeric(matrix(w, nrow = 1) %*% x))
}
