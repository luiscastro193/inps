import inps
import pandas as pd
import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from xgboost import XGBRegressor, XGBClassifier

np_sample = pd.read_parquet("./data/ED/nonprobEd.parquet")
p_sample = pd.read_parquet("./data/ED/probEd.parquet")
population_size = 40884010

psa_covariates = ["Altura", "Peso", "Estudios", "Frecuencia_acf", "IMC_rec"]
calibration_covariates = ["Sexo", "Edad_rec"]

levels = {
	"Estudios": ('Educación general básica u obligatoria (primaria, ESO)', 'Grado medio o Formación Profesional I (FP1)', 'Bachillerato, Grado Superior o Formación Profesional II (FP2)', 'Estudios Universitarios (Grado, Diplomatura o Licenciatura)', 'Estudios de Postgrado (Máster, Doctorado)'),
	"Frecuencia_acf": ('Rara vez', '1 vez por semana', '2-3 veces por semana', 'Casi todos los días'),
	"IMC_rec": ('Infrapeso', 'Normopeso', 'Sobrepeso', 'Obesidad')
}

for sample in np_sample, p_sample:
	for column, ordered_values in levels.items():
		sample[column] = sample[column].cat.rename_categories(lambda code: ordered_values.index(code)).astype("Int64")

psa_weights = inps.psa_weights(np_sample, p_sample, population_size, covariates = psa_covariates)
kw_weights = inps.kw_weights(np_sample, p_sample, population_size, covariates = psa_covariates)

psa_weights2 = inps.psa_weights(np_sample, p_sample, population_size, covariates = psa_covariates, model = inps.make_preprocess_estimator(BernoulliNB()))
kw_weights2 = inps.kw_weights(np_sample, p_sample, population_size, covariates = psa_covariates, model = XGBClassifier())

target_var = "D4"
naive_estimation = np.average(np_sample[target_var])
psa_estimation = np.average(np_sample[target_var], weights = psa_weights["np"])
kw_estimation = np.average(np_sample[target_var], weights = kw_weights)

print(naive_estimation, psa_estimation, kw_estimation)

population_totals = pd.Series({'Hombre': 19972404, 'Mujer': 20911606, '18-29': 7406234, '30-44': 9691504, '45+': 23786272})

np_sample["Edad_rec"] = np_sample["Edad_rec"].cat.rename_categories({'1': '18-29', '2': '30-44', '3': '45+'})
np_sample = pd.get_dummies(np_sample, prefix = '', prefix_sep = '', columns = calibration_covariates)

np_sample["weights"] = psa_weights["np"]
psa_calibration_weights = inps.calibration_weights(np_sample, population_totals, weights_column = "weights")
psa_calibration_estimation = np.average(np_sample[target_var], weights = psa_calibration_weights)

print(psa_calibration_estimation)

np_sample = pd.read_parquet("./data/covid/nonprobCovid.parquet")
p_sample = pd.read_parquet("./data/covid/probCovid.parquet")

covariates = ['age', 'gender', 'region', 'household_size', 'employment_status']

p_covariates_names = ['EDAD', 'SEXO', 'CCAA', 'NUMHOGAR', 'SITLAB']
p_sample.rename(columns = dict(zip(p_covariates_names, covariates)), inplace = True)

p_sample['gender'] = p_sample['gender'].replace({1: "Male", 2: "Female"})

regions = ('', 'Andalucía', 'Aragón', 'Principado de Asturias', 'Islas Baleares', 'Islas Canarias', 'Cantabria', 'Castilla-La Mancha', 'Castilla y León', 'Cataluña', 'Comunidad Valenciana', 'Extremadura', 'Galicia', 'Comunidad de Madrid', 'Región de Murcia', 'Comunidad Foral de Navarra', 'País Vasco', 'La Rioja', 'Ceuta', 'Melilla')
p_sample['region'] = p_sample['region'].map(lambda code: regions[code], na_action = 'ignore')

np_sample['household_size'] = pd.to_numeric(np_sample['household_size'].replace({'8 or more': 8}), errors = 'coerce')
p_sample['household_size'] = p_sample['household_size'].replace(99, np.nan)
p_sample.loc[p_sample['household_size'] > 8, 'household_size'] = 8

np_sample['employment_status'] = np_sample['employment_status'].replace({
	'Full time employment': 'Employed', 'Part time employment': 'Employed', 'Full time student': 'Student', 'Retired': 'Unemployed', 'Not working': 'Other'
})
p_status = ('', 'Employed', 'Unemployed', 'Unemployed', 'Unemployed', 'Unemployed', 'Student', 'Other', 'Other', np.nan)
p_sample['employment_status'] = p_sample['employment_status'].map(lambda code: p_status[code], na_action = 'ignore')

for categorical in ('gender', 'region', 'employment_status'):
	for sample in np_sample, p_sample:
		sample[categorical] = sample[categorical].astype('category')

categories = ('Not at all', 'Rarely', 'Sometimes', 'Frequently', 'Always')
np_sample['target_num'] = np_sample['i12_health_7'].map(lambda code: categories.index(code), na_action = 'ignore') + 1

matching_values = inps.matching_values(np_sample, p_sample, "target_num", covariates = covariates, model = inps.make_preprocess_estimator(MLPRegressor()))
training_values = inps.training_values(np_sample, p_sample, "target_num", covariates = covariates, psa_model = XGBClassifier(enable_categorical = True, tree_method = "hist"), matching_model = XGBRegressor(enable_categorical = True, tree_method = "hist"))

naive_estimation = np.average(np_sample['target_num'])
matching_estimation = np.average(matching_values['p'])
training_estimation = np.average(training_values['p'])

dr_estimation = inps.doubly_robust_estimation(np_sample, p_sample, "target_num", covariates = covariates, matching_model = inps.make_preprocess_estimator(KNeighborsRegressor()))

print(naive_estimation, matching_estimation, training_estimation, dr_estimation)

np_sample['target_cat'] = np_sample['target_num'] > 3
training_values = inps.training_values(np_sample, p_sample, "target_cat", True, covariates = covariates)

naive_estimation = np.average(np_sample['target_cat'])
training_estimation = np.average(training_values['p'])

print(naive_estimation, training_estimation)

np_sample = pd.read_parquet("./data/health/nonprobHealth.parquet")
population = pd.read_parquet("./data/health/censusHealth.parquet")
covariates = ["sexo", "b2", "dapae_memo", "PERFILPROFTITUL"]
target_var = "CRONICIDAD"
target_category = "Sí"

np_sample = np_sample[np_sample[target_var].notna()]

imputed_values = inps.training_values(np_sample, population, target_var, target_category, covariates = covariates)
imputed_values2 = inps.matching_values(np_sample, population, target_var, target_category, covariates = covariates, model = inps.make_preprocess_estimator(KNeighborsClassifier()))

naive_estimation = np.average(np_sample[target_var] == target_category)
model_estimation = np.average(imputed_values["p"])
custom_model_estimation = np.average(imputed_values2["p"])

print(naive_estimation, model_estimation, custom_model_estimation)
