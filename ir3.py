import numpy as np
import pandas as pd
from pgmpy.models.DiscreteBayesianNetwork import DiscreteBayesianNetwork as BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# load the dataset

heartDisease = pd.read_csv('/content/heart.csv')
display(heartDisease.head())

heartDisease = heartDisease.replace('?',np.nan)
print(f"Few examples from the dataset are given below : \n\n{heartDisease.head()}")

# Model Bayesian Network
model = BayesianModel([('age','trestbps'),('age','fbs'),('sex','trestbps'),('exang','trestbps'),('trestbps','heartdisease'),
                       ('fbs','heartdisease'),('heartdisease','restecg'), ('heartdisease','thalach'),('heartdisease','chol')])


# Learning CPDs using Maximum Likelihood Estimators
print('\nLearning CPD using Maximum likelihood estimators')
model.fit(heartDisease, estimator=MaximumLikelihoodEstimator)

# Inferencing with Bayesian Network
print('Inferencing with Bayesian Network:')
HeartDisease_infer = VariableElimination(model)

# Computing the Probability of HeartDisease given Age
print('1. Probability of HeartDisease given Age=38')
q = HeartDisease_infer.query(variables=['heartdisease'], evidence={'age':38})
print(q)

# Computing the Probability of HeartDisease given cholesterol
print('\n 2. Probability of HeartDisease given cholesterol=230')
q=HeartDisease_infer.query(variables=['heartdisease'], evidence ={'chol':230})
print(q)
