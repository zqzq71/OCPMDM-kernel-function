#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 1/4/18 10:52 AM
# @Author  : Qing Zhang
# @Site    : chemdata.shu.edu.cn
# @File    : material_splitter.py
import sys
sys.path.append('../')

import re
import os
import math

class material_splitter():
    def __init__(self, element_file_path='./element_data/CommonData.txt'):
        # load common_file
        self._element_list = material_splitter.load_element_list(element_file_path)

    def split_material(self, type, src_str_list):

        if type.lower() == 'perovskite':
            return list(map(self._split_perovskite_formula, src_str_list))


    def _split_perovskite_formula(self, perovskite_string):

        buff_list = list(perovskite_string)
        digit_flag = [d.isdigit() or d == '.' for d in buff_list]

        element_part = []
        weight_part = []

        cache = []
        before_flag = False

        for flag, str_ele in zip(digit_flag, buff_list):
            if before_flag == flag:
                cache.append(str_ele)
            else:
                if before_flag is False:
                    upper_lower_flag = [d.islower() for d in cache]

                    before_char = []

                    for index, (f, char) in enumerate(zip(upper_lower_flag, cache)):
                        before_char.append(char)
                        if f:
                            element_part.append(''.join(before_char))
                            before_char = []
                            if index < len(upper_lower_flag) - 1:
                                weight_part.append('1.0')
                    if len(before_char) > 0:
                        element_part.append(''.join(before_char))
                else:
                    weight_part.append(''.join(cache))

                before_flag = flag
                cache = [str_ele]

        if before_flag is False:
            upper_lower_flag = [d.islower() for d in cache]

            before_char = []

            for index, (f, char) in enumerate(zip(upper_lower_flag, cache)):
                before_char.append(char)
                if f:
                    element_part.append(''.join(before_char))
                    before_char = []
                    if index < len(upper_lower_flag) - 1:
                        weight_part.append('1.0')
            if len(before_char) > 0:
                element_part.append(''.join(before_char))
        else:
            weight_part.append(''.join(cache))

        # check element
        if any(map(lambda x: x not in self._element_list, element_part)):
            return 'error element'

        # check last element is O
        if element_part[-1] != 'O':
            return 'the last element is not O please check'

        element_part = element_part[:-1]
        weight_part = [float(d) for d in weight_part[:-1]]

        sum_weight = sum(weight_part)

        # split by weight
        for i in range(len(weight_part)):
            if math.isclose(sum(weight_part[: i + 1]), sum_weight / 2):
                cut_point = i + 1

        element_a = element_part[: cut_point]
        element_b = element_part[cut_point:]

        weight_a = weight_part[: cut_point]
        weight_b = weight_part[cut_point:]

        # normalize weight
        weight_a = [d / (sum_weight / 2) for d in weight_a]
        weight_b = [d / (sum_weight / 2) for d in weight_b]

        part_a = ','.join(['{},{}'.format(e, w) for e, w in zip(element_a, weight_a)])
        part_b = ','.join(['{},{}'.format(e, w) for e, w in zip(element_b, weight_b)])

        return [part_a, part_b]

    @staticmethod
    def load_element_list(file_name):
        element_list = []
        with open(file_name, 'r') as reader:
            for line in reader.readlines():
                element_list.append(line.split('\t')[0])
        return element_list