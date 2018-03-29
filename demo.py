
import pandas as pd
import numpy as np

from descriptor_filler.material_descriptor import material_descriptor
from rvm.rvm import RVM
from rvm.Normalize import *
from virtual_screen.ga_virtual_screening import *

class modified_rvm():
    def __init__(self, task_info, feature_index):
        self._model = RVM(task_info)
        self._feature_index = feature_index

    def fit(self, X, y):
        X = X[:, self._feature_index]
        self._model.fit(X, y)

    def predict(self, X):
        return self._model.predict(X[:, self._feature_index])

if __name__ == '__main__':

    # Filling material descriptors

    perovsite_descriptor_generator = material_descriptor('perovskite', './element_data/CommonData.txt')
    perovsite_descriptor_generator.fill('./', 'source_data.csv')

    print('Filling material descriptors and save the data to "Filled_Descriptor.csv"')

    # use RVM build regressor
    print('Load data and use RVM to build model')

    df = pd.read_csv('./Filled_Descriptor.csv')
    data = df.values

    src_x, src_y = data[:, 3: ], data[:, 2]
    normalizer = FeatureScaler('normalize')
    train_x = normalizer(src_x)
    scaler = TargetMapping()
    train_y = scaler(src_y)

    material_model = modified_rvm({'Kernel':'+gauss','Kernel_Param':'0.9','MaxIts':'10'},
                                  feature_index=[2, 6, 8, 11, 16, 17])
    material_model.fit(train_x, train_y)

    # do screening candidate material with higher property
    print('Screening candidate material with higher property...')

    mat_fitness = material_screen_fitness((material_model, normalizer, scaler),
                                          perovsite_descriptor_generator,
                                          {"A": [{"weight": "0.6, 1.0, 0.02", "element": "La"},
                                                 {"weight": "0.0, 0.4, 0.02", "element": "Ca,Sr,Ba"},
                                                 {"weight": "None", "element": "Ca,Sr,Ba,Nd,Dy,Y,Mg,Eu"}],
                                           "B": [{"weight": "0.9, 1.0, 0.02", "element": "Mn"},
                                                 {"weight": "None", "element": "Fe,V,Ti,Zn,Cr"}]})



    ga_obj = ga_material_finder({'NGEN': 20,
                                 'population': 80,
                                 'CXPB': 0.5},
                                mat_fitness, './', True)
    result_material = ga_obj.fit()

    ### output result ###
    perovskite_candidate_material = result_material[0].replace(',', '') + 'O3'
    print('Perovskite material candidate: {}, Cuire temperature is: {:.4g}K.'.format(perovskite_candidate_material, result_material[1]))


