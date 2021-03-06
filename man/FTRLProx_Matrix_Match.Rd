% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/FTRLProx_Matrix_Match.R
\name{FTRLProx_Matrix_Match}
\alias{FTRLProx_Matrix_Match}
\title{Transform Sparse Design Matrix to Assigned Schema}
\usage{
FTRLProx_Matrix_Match(data, feature_name)
}
\arguments{
\item{data}{a object of \code{ftrl.Dataset}.}

\item{feature_name}{the feature name of design matrix}
}
\value{
constructed dataset, an object of class "ftrl.Dataset"
}
\description{
Transform a sparse "design" matrix to assigned schema for FTRL-Proximal Algorithm.
}
\examples{
library(data.table)
library(FeatureHashing)
data(ipinyou)
f <- ~ 0 + BidID + IP + City + AdExchange + BiddingPrice + PayingPrice
m.train <- FTRLProx_Model_Matrix(f, ipinyou.train[, -"IsClick", with = FALSE])
m.test <- FTRLProx_Model_Matrix(f, ipinyou.test[, -"IsClick", with = FALSE])
identical(rownames(m.train$X), rownames(m.test$X))
m.test <- FTRLProx_Matrix_Match(m.test, rownames(m.train$X))
identical(rownames(m.train$X), rownames(m.test$X))
m.test$Mapping
}
