"""Python package for statistical inference from non-probability samples"""

__version__ = "1.1.3"

from math import sqrt
import numpy as np
import pandas as pd
import sklearn
from pandas.api.types import is_numeric_dtype
from scipy.stats import iqr
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as column_selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import RidgeCV, LogisticRegressionCV

sklearn.set_config(enable_metadata_routing = True)

def default_preprocess(**kwargs):
	return ColumnTransformer([
		("numeric", Pipeline([
			("normalizer", RobustScaler()),
			("imputer", SimpleImputer(strategy = 'median', add_indicator = True, copy = False))
		]), column_selector(dtype_include = 'number')),
		("categorical",
			OneHotEncoder(drop = 'if_binary', min_frequency = .05, handle_unknown = 'infrequent_if_exist'),
		column_selector(dtype_exclude = 'number'))
	], **kwargs)

def make_preprocess_estimator(base_estimator, **kwargs):
	if hasattr(base_estimator, "set_fit_request"):
		base_estimator = base_estimator.set_fit_request(sample_weight = True)
	if hasattr(base_estimator, "set_score_request"):
		base_estimator = base_estimator.set_score_request(sample_weight = True)
	
	return Pipeline([
		("preprocess", default_preprocess(**kwargs)),
		("estimator", base_estimator)
	])

def logistic_classifier(**kwargs):
	return make_preprocess_estimator(LogisticRegressionCV(cv = StratifiedKFold(shuffle = True, random_state = 0), scoring = 'neg_log_loss', max_iter = 1000, **kwargs))

def linear_regressor(**kwargs):
	return make_preprocess_estimator(RidgeCV(**kwargs))

def calibration_weights(sample, population_totals, weights_column = None, population_size = None, max_steps = 1000, tolerance = 1e-6):
	X = sample.loc[:, population_totals.index].to_numpy(dtype = 'float64')
	
	if weights_column is not None:
		d = sample[weights_column].to_numpy()
		X = X * d.reshape(-1, 1)
	elif population_size is not None:
		d = population_size / X.shape[0]
		X = X * d
	else:
		raise ValueError("weights_column or population_size must be set")
	
	T = population_totals.to_numpy()
	L = np.zeros(X.shape[1])
	w = np.ones(X.shape[0])
	H = np.eye(X.shape[0])
	success = False
	
	for step in range(max_steps):
		L += np.dot(np.linalg.pinv(np.dot(np.dot(X.T, H), X)), (T - np.dot(X.T, w)))
		w = np.exp(np.dot(X, L))
		
		loss = np.max(np.abs(np.dot(X.T, w) - T) / T)
		if loss < tolerance:
			success = True
			break
		
		H = np.diag(w)
	
	if not success: raise Exception("Calibration did not converge")
	return w * d

def propensities(np_sample, p_sample, weights_column = None, covariates = None, model = None):
	np_size = np_sample.shape[0]
	p_size = p_sample.shape[0]
	weights = np.ones(p_size) if weights_column is None else p_sample[weights_column]
	
	if covariates is not None:
		np_sample = np_sample.loc[:, covariates]
		p_sample = p_sample.loc[:, covariates]
	
	if model is None: model = logistic_classifier()
	
	X = pd.concat((np_sample, p_sample), ignore_index = True, join = "inner", copy = False)
	y = np.concatenate((np.ones(np_size, dtype = bool), np.zeros(p_size, dtype = bool)))
	sample_weight = np.concatenate((np.repeat(np.sum(weights) / np_size, np_size), weights))
	sample_weight /= np.mean(sample_weight)
	model.fit(X, y, sample_weight = sample_weight)
	return model.predict_proba(X)[:, tuple(model.classes_).index(True)]

def psa_weights(np_sample, p_sample, population_size = None, weights_column = None, covariates = None, model = None):
	np_size = np_sample.shape[0]
	my_propensities = propensities(np_sample, p_sample, weights_column, covariates, model)
	optimal_weights = (1 - my_propensities) / my_propensities
	
	if population_size:
		optimal_weights *= population_size / np.sum(optimal_weights[:np_size])
	
	return {"np": optimal_weights[:np_size], "p": optimal_weights[np_size:]}

def matching_values(np_sample, p_sample, target_column, target_category = None, covariates = None, model = None, training_weight = None):
	y = np_sample[target_column]
	
	if y.isna().any(): raise ValueError("Missing values in target column")
	
	if not is_numeric_dtype(y) and target_category is None:
		raise ValueError("target_category must be set when the target variable is categorical.")
	
	if target_category is not None: y = y == target_category
	
	if model is None:
		model = linear_regressor() if target_category is None else logistic_classifier()
	
	if covariates is None:
		covariates = np_sample.columns.intersection(p_sample.columns)
	
	np_sample = np_sample.loc[:, covariates]
	p_sample = p_sample.loc[:, covariates]
	
	if training_weight is None:
		model.fit(np_sample, y)
	else:
		training_weight /= np.mean(training_weight)
		model.fit(np_sample, y, sample_weight = training_weight)
	
	def predict(X):
		if target_category is None:
			return model.predict(X)
		else:
			return model.predict_proba(X)[:, tuple(model.classes_).index(True)]
	
	return {"p": predict(p_sample), "np": predict(np_sample)}

def doubly_robust_estimation(np_sample, p_sample, target_column, target_category = None, weights_column = None, covariates = None, psa_model = None, matching_model = None):
	if weights_column is None:
		weights = None
	else:
		weights = p_sample[weights_column]
	
	imputed_weights = psa_weights(np_sample, p_sample, None, weights_column, covariates, psa_model)["np"]
	imputed_values = matching_values(np_sample, p_sample, target_column, target_category, covariates, matching_model)
	original_values = np_sample[target_column] if target_category is None else np_sample[target_column] == target_category
	return np.average(imputed_values["p"], weights = weights) + np.average(original_values - imputed_values["np"], weights = imputed_weights)

def training_values(np_sample, p_sample, target_column, target_category = None, weights_column = None, covariates = None, psa_model = None, matching_model = None):
	training_weight = psa_weights(np_sample, p_sample, None, weights_column, covariates, psa_model)["np"]
	return matching_values(np_sample, p_sample, target_column, target_category, covariates, matching_model, training_weight)

def kw_weights(np_sample, p_sample, population_size = None, weights_column = None, covariates = None, model = None):
	np_size = np_sample.shape[0]
	my_propensities = propensities(np_sample, p_sample, weights_column, covariates, model)
	
	distances = my_propensities[np_size:].reshape(-1, 1) - my_propensities[:np_size]
	m = min(sqrt(np.var(distances)), iqr(distances) / 1.349)
	h = 0.9 * m / pow(distances.size, 1/5)
	kernels = np.exp(-distances**2 / (2*h*h))
	kernels[kernels == 0] = np.min(kernels[kernels != 0])
	kernels /= np.sum(kernels, axis = 1).reshape(-1, 1)
	
	if weights_column is None:
		kernels = np.sum(kernels, axis = 0)
	else:
		kernels = np.sum(kernels * p_sample[weights_column].to_numpy().reshape(-1, 1), axis = 0)
	
	if population_size: kernels *= population_size / np.sum(kernels)
	return kernels
