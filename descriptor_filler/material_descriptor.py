#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 5/9/17 12:02 AM
# @Author  : Qing Zhang
# @Site    : chemdata.shu.edu.cn
# @File    : material_descriptor.py
# import sys
# sys.path.append('../')

from .material_splitter import material_splitter
import numpy as np
import pandas as pd
import os

class material_descriptor():
    def __init__(self, data_type, element_file_path='./element_data/CommonData.txt'):
        self._data_type = data_type
        self._elementcomment = material_descriptor.load_CommonData(element_file_path)

    @property
    def elements(self):
        return self._elementcomment


    def fill(self, file_path, file_name):
        if self._data_type == 'perovskite':
            return self._perovskite_data(file_path, file_name)

    def _perovskite_data(self, file_path, file_name):
        target_name = ''
        other_feature_names = []
        target_column = []
        formula_column = []
        other_feature_columns = []

        with open(os.path.join(file_path, file_name), 'r') as reader:
            all_line = reader.readlines()

            title_cache = all_line[0][:-1].split(',')
            target_name = title_cache[0]
            other_feature_names = title_cache[2: ]

            for line in all_line[1: ]:
                buff = line[:-1].split(',')
                target_column.append(buff[0])
                formula_column.append(buff[1])
                other_feature_columns.append(buff[2: ])

        splitter = material_splitter()
        splitted_formula = splitter.split_material('perovskite', formula_column)

        material_atom_descriptor = material_descriptor.fill_perovskite_features(splitted_formula, self._elementcomment)
        material_atom_descriptor_names = material_atom_descriptor[0]

        title_line = ['Number', 'Class'] + [target_name] + material_atom_descriptor_names + other_feature_names
        number_column = np.array(list(range(1, len(material_atom_descriptor))))
        class_column = np.array([0] * (len(material_atom_descriptor) - 1))
        target_column = np.array([float(d) for d in target_column])

        good_index = np.where(target_column >= np.mean(target_column))[0]
        class_column[good_index] = 1

        descriptor_data = np.array(material_atom_descriptor[1: ])
        other_feature_columns = np.array(other_feature_columns)
        if len(other_feature_columns.shape) < 2:
            other_feature_columns = other_feature_columns[:, np.newaxis]

        all_data = np.hstack((number_column[:, np.newaxis], class_column[:, np.newaxis], target_column[:, np.newaxis], descriptor_data, other_feature_columns))
        df = pd.DataFrame(all_data, columns=title_line)
        df.to_csv(os.path.join(file_path, 'Filled_Descriptor.csv'), index=None)

        return 'Filled_Descriptor.csv'

    @staticmethod
    def get_material_feature_title(type):
        if type.lower() == 'perovskite':
            return ['Radius_A', 'Radius_B', 'Za', 'Zb',
                    'TF', 'aO3', 'rc', 'A_ionic', 'B_ionic',
                    'R_a/R_b', 'Mass', 'A_aff','B_aff', 'A_Tm',
                    'B_Tm', 'A_Tb', 'B_Tb', 'A_Hfus', 'B_Hfus',
                    'A_Density', 'B_Density']

    @staticmethod
    def load_CommonData(filename):
        result = {}

        fread = open(filename, 'r')

        for line in fread.readlines():
            line = line[:-1]
            buf = line.split('\t')
            result[buf[0].strip()] = buf[1:]
        fread.close()

        return result

    @staticmethod
    def fill_perovskite_features(srcContent, elementCommennt, have_title=True):
        output_list = []
        table_header_list = ['Radius_A', 'Radius_B', 'Za', 'Zb',
                             'TF', 'aO3', 'rc', 'A_ionic', 'B_ionic', 'R_a/R_b', 'Mass', 'A_aff',
                             'B_aff', 'A_Tm', 'B_Tm', 'A_Tb', 'B_Tb', 'A_Hfus', 'B_Hfus', 'A_Density', 'B_Density']

        if have_title:
            output_list.append(table_header_list)

        for i in range(len(srcContent)):
            # check split result
            if type(srcContent[i]) is not list:
                output_list.append(['None'] * len(table_header_list))
                continue

            # a part
            aradiusbuf = srcContent[i][0].split(',')

            aradiuslen = int(len(aradiusbuf) / 2)

            a_radius = 0
            Za = material_descriptor.getAverageProperty(aradiusbuf, elementCommennt, 3)
            AMW = material_descriptor.getAverageProperty(aradiusbuf, elementCommennt, 0)
            A_aff = material_descriptor.getAverageProperty(aradiusbuf, elementCommennt, 8)
            A_ionic = material_descriptor.getAverageProperty(aradiusbuf, elementCommennt, 11)
            A_tm = material_descriptor.getAverageProperty(aradiusbuf, elementCommennt, 12)
            A_tb = material_descriptor.getAverageProperty(aradiusbuf, elementCommennt, 13)
            A_hfus = material_descriptor.getAverageProperty(aradiusbuf, elementCommennt, 14)
            A_density = material_descriptor.getAverageProperty(aradiusbuf, elementCommennt, 15)

            for j in range(aradiuslen):
                # ra
                a_part_weight = float(aradiusbuf[2 * j + 1])
                a_part_element = aradiusbuf[2 * j]
                a_part_radius = material_descriptor.selectRadius(a_part_element, elementCommennt)

                if a_part_radius == 'None':
                    a_radius = 'None'
                    break
                a_radius += (a_part_radius * a_part_weight)

            # b part
            bradiusbuf = srcContent[i][1].split(',')
            bradiuslen = int(len(bradiusbuf) / 2)

            b_radius = 0
            Zb = material_descriptor.getAverageProperty(bradiusbuf, elementCommennt, 3)
            BMW = material_descriptor.getAverageProperty(bradiusbuf, elementCommennt, 0)
            B_aff = material_descriptor.getAverageProperty(bradiusbuf, elementCommennt, 8)
            B_ionic = material_descriptor.getAverageProperty(bradiusbuf, elementCommennt, 11)
            B_tm = material_descriptor.getAverageProperty(bradiusbuf, elementCommennt, 12)
            B_tb = material_descriptor.getAverageProperty(bradiusbuf, elementCommennt, 13)
            B_hfus = material_descriptor.getAverageProperty(bradiusbuf, elementCommennt, 14)
            B_density = material_descriptor.getAverageProperty(bradiusbuf, elementCommennt, 15)

            for j in range(bradiuslen):
                # rb
                b_part_weight = float(bradiusbuf[2 * j + 1])
                b_part_element = bradiusbuf[2 * j]
                b_part_radius = material_descriptor.selectRadius(b_part_element, elementCommennt)

                if b_part_radius == 'None':
                    b_radius = 'None'
                    break
                b_radius += (b_part_radius * b_part_weight)

            TF = 'None'

            if a_radius != 'None' and b_radius != 'None':
                TF = (a_radius + 140.0) / (np.sqrt(2) * (b_radius + 140.0))

            aO3 = 'None'

            if b_radius != 'None' and TF != 'None':
                aO3 = 2.37 * b_radius + 2.47 - 2 * ((1 / TF) - 1)

            rc = 'None'

            if a_radius != 'None' and b_radius != 'None' and aO3 != 'None':
                rc = (a_radius * a_radius + float(3 / 4) * aO3 * aO3 - np.sqrt(
                    2) * aO3 * b_radius + b_radius * b_radius) / (2 * a_radius + np.sqrt(2) * aO3 - float(2) * b_radius)

            ra_rb = 'None'

            if a_radius != 'None' and b_radius != 'None':
                try:
                    ra_rb = (a_radius) / (b_radius)
                except:
                    ra_rb = 'None'

            za_zb = 'None'

            All_M = 'None'

            if AMW != 'None' and BMW != 'None':
                All_M = AMW + BMW + 48.0

            result_list = [a_radius, b_radius, Za, Zb, TF, aO3, rc,
                           A_ionic, B_ionic, ra_rb, All_M, A_aff, B_aff, A_tm,
                           B_tm, A_tb, B_tb, A_hfus, B_hfus, A_density, B_density]
            output_list.append(result_list)
        return output_list

    @staticmethod
    def selectRadius(element, elementCommon):

        # firstly find +3 radius
        radiusBuf = elementCommon[element][6]

        if radiusBuf != 'None':
            return float(radiusBuf)

        radiusBuf = elementCommon[element][5]

        if radiusBuf != 'None':
            return float(radiusBuf)

        return 'None'
    @staticmethod
    def getAverageProperty(src_data, elementCommon, property_index):
        src_data_len = int(len(src_data) / 2)

        result = 0

        for j in range(src_data_len):
            part_property = elementCommon[src_data[2 * j]][property_index].strip()
            part_weigth = float(src_data[2 * j + 1])

            if part_property != 'None':
                part_property = float(part_property)
                result += (part_property * part_weigth)
            else:
                result = 'None'
                break
        return result
