# -*- coding: utf-8 -*-
"""
Created on Sat May  1 20:49:49 2021

@author: RISHBANS
"""

import pandas as pd
from sklearn.datasets import load_iris
iris = load_iris()
iris.feature_names
iris.target_names

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['type'] = iris.target

import numpy as np
class MyKNN:
    def __init__(self,k=5):
        self.k = k
    
    #convert and assign feature and target data to variable   
    def my_fit(self, feature_data, target_data):
        self.feature_data = np.array(feature_data)
        self.target_data = np.array(target_data)
     
    #Calculate Euclidean Distance, axis-> 1: Column, 0:row    
    #[3.5, 2.5, 3.4, 2.3] - [1,2,3,4] - like this for all rows
    def calculate_distance_vector_matrix(self, one_data):
        distances = np.sqrt(np.sum(np.square(self.feature_data - one_data),axis=1))
        return distances
    
    #Sort the distance and take top k=5 which are closest
    def find_k_neighbours(self,one_data_feature):
        res = self.calculate_distance_vector_matrix(one_data_feature)
        return res.argsort()[:self.k]
        
    def find_k_neighbours_class(self, one_data_feature):
        indexs_of_neighbours = self.find_k_neighbours(one_data_feature)
        return self.target_data[indexs_of_neighbours]
    
    def my_predict(self, one_data_feature):
        classes = self.find_k_neighbours_class(one_data_feature)
        return np.bincount(classes).argmax()
    
model = MyKNN(k=5)
df.columns
feature_data = df.drop(columns=['type'],axis=1)
target_data = df.type
model.my_fit(feature_data, target_data)

one_data = [1,2,3,4]
model.find_k_neighbours_class(one_data)
print(model.my_predict([1,2,3,4]))
