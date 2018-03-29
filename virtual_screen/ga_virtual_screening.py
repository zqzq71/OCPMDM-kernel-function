import numpy as np
import pandas as pd
import os
import random

from deap import base
from deap import creator
from deap import tools

class material_screen_fitness():
    def __init__(self, model, material_descriptor, virtual_screen_setting):

        self.model, self.normalizer, self.scaler = model
        self.material_candidate = virtual_screen_setting

        self._material_descriptor = material_descriptor

        self._param_count = 0
        self.ele_A, candidate_A_weights = self._get_info(self.material_candidate['A'])
        self.ele_B, candidate_B_weights = self._get_info(self.material_candidate['B'])

        self.w_A = self._generate_candidate_weight(candidate_A_weights)
        self.w_B = self._generate_candidate_weight(candidate_B_weights)
        self.w_A = self.w_A[:-1]
        self.w_B = self.w_B[:-1]
        self._generate_evalue_param()

    @property
    def param_count(self):
        return self._param_count

    def _generate_candidate_weight(self, weight_list):
        candidate_weight_list = []

        for ele in weight_list:
            if ele == 'None':
                candidate_weight_list.append([])
            else:
                candidate_weight_list.append(np.arange(ele[0], ele[1] + ele[2], ele[2]))

        return candidate_weight_list

    def _get_info(self, site_info):

        element_list = []
        weight_list = []

        for obj in site_info:
            element_list.append(obj['element'].split(','))
            if obj['weight'] != 'None':
                weight_list.append([float(d) for d in obj['weight'].split(',')])
            else:
                weight_list.append('None')

        return element_list, weight_list

    def _generate_evalue_param(self):
         self._param_count = sum([len(self.ele_A), len(self.ele_B), len(self.w_A), len(self.w_B)])

    def convert_param(self, param_list):

        result = []
        for i, d in enumerate(self._min_max_list):
            min_val, max_val = d[1:]
            result.append(param_list[i] * (max_val - min_val) + min_val)

        return result
    def _map_value(self, src_list, value):
        interval = 1. / float(len(src_list))
        for i in range(len(src_list)):
            if i < value / interval <= (i + 1):
                return src_list[i]

    def _convert_individual_to_content(self, individual, src_list, is_fill=False):
        return_list = []
        for v, candidate_list in zip(individual, src_list):
            return_list.append(self._map_value(candidate_list, v))

        if is_fill:
            return_list = [float(d) for d in return_list]
            if np.sum(return_list) <= 1.:
                return_list.append(1. - np.sum(return_list))
            else:
                return 'out of range'

        return return_list

    def recovery_material(self, individual):
        current_pt = 0
        current_pt += len(self.ele_A)
        ele_A_part = individual[: current_pt]
        ele_B_part = individual[current_pt: current_pt + len(self.ele_B)]
        current_pt += len(self.ele_B)
        w_A_part = individual[current_pt: current_pt + len(self.w_A)]
        current_pt += len(self.w_A)
        w_B_part = individual[current_pt:]

        processed_ele_A = self._convert_individual_to_content(ele_A_part, self.ele_A)
        processed_ele_B = self._convert_individual_to_content(ele_B_part, self.ele_B)
        processed_w_A = self._convert_individual_to_content(w_A_part, self.w_A, True)
        processed_w_B = self._convert_individual_to_content(w_B_part, self.w_B, True)

        if processed_w_A == 'out of range' or processed_w_B == 'out of range':
            return 0.0,

        str_a = ','.join(['{},{}'.format(e, w) for e, w in zip(processed_ele_A, processed_w_A)])
        str_b = ','.join(['{},{}'.format(e, w) for e, w in zip(processed_ele_B, processed_w_B)])

        current_virtual_material_data = self._material_descriptor.fill_perovskite_features([[str_a, str_b]],
                                                                                            self._material_descriptor.elements,
                                                                                            have_title=False)

        current_virtual_material_data = np.array([0] + current_virtual_material_data[0], dtype='float32')[
                                        np.newaxis, 1:]

        if not self.normalizer is None:
            current_virtual_material_data = self.normalizer.transform(current_virtual_material_data[0])

        predicted = self.model.predict(current_virtual_material_data[np.newaxis, :])

        if not self.scaler is None:
            predicted = self.scaler.recovery(predicted)

        return '{}{}'.format(str_a, str_b), predicted[0]

    def fitness(self, individual):

        current_pt = 0
        current_pt += len(self.ele_A)
        ele_A_part = individual[: current_pt]
        ele_B_part = individual[current_pt: current_pt + len(self.ele_B)]
        current_pt += len(self.ele_B)
        w_A_part = individual[current_pt: current_pt + len(self.w_A)]
        current_pt += len(self.w_A)
        w_B_part = individual[current_pt: ]

        processed_ele_A = self._convert_individual_to_content(ele_A_part, self.ele_A)
        processed_ele_B = self._convert_individual_to_content(ele_B_part, self.ele_B)
        processed_w_A = self._convert_individual_to_content(w_A_part, self.w_A, True)
        processed_w_B = self._convert_individual_to_content(w_B_part, self.w_B, True)

        if processed_w_A == 'out of range' or processed_w_B == 'out of range':
            return 0.0,

        str_a = ','.join(['{},{}'.format(e, w) for e, w in zip(processed_ele_A, processed_w_A)])
        str_b = ','.join(['{},{}'.format(e, w) for e, w in zip(processed_ele_B, processed_w_B)])

        current_virtual_material_data = self._material_descriptor.fill_perovskite_features([[str_a, str_b]],
                                                                                           self._material_descriptor.elements,
                                                                                           have_title=False)
        current_virtual_material_data = [0] + current_virtual_material_data[0]

        if 'None' in current_virtual_material_data:
            return 0.0,

        current_virtual_material_data = np.array(current_virtual_material_data, dtype='float32')[
            np.newaxis, 1: ]

        if not self.normalizer is None:
            current_virtual_material_data = self.normalizer.transform(current_virtual_material_data[0])

        predicted = self.model.predict(current_virtual_material_data[np.newaxis, :])

        if not self.scaler is None:
            predicted = self.scaler.recovery(predicted)

        return predicted[0],


class ga_material_finder():
    def __init__(self, task_info, fitness_obj, file_path, show_opts_history=True):
        self._file_path = file_path
        self._task_info = task_info
        self._fitness_obj = fitness_obj
        self._show_opts_history = show_opts_history

        # storage result
        self._ga_results_list = []

    @property
    def task_info(self):
        return self._task_info

    def _uniform(self, low, up, size=None):
        try:
            return [random.uniform(a, b) for a, b in zip(low, up)]
        except TypeError:
            return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]

    def fit(self):
        creator.create('FitnessMax', base.Fitness, weights=(1.0,))
        creator.create('Individual', list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()

        BOUND_LOW, BOUND_UP = 0.000001, 1.0

        NDIM = self._fitness_obj.param_count

        toolbox.register('attr_float', self._uniform, BOUND_LOW, BOUND_UP, NDIM)
        toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.attr_float)
        toolbox.register('population', tools.initRepeat, list, toolbox.individual)

        toolbox.register('evaluate', self._fitness_obj.fitness)
        toolbox.register('mate', tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
        toolbox.register('mutate', tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0 / NDIM)
        toolbox.register("select", tools.selNSGA2)

        NGEN = int(self._task_info['NGEN']) + 1
        pop_size = int(self._task_info['population'])
        CXPB = float(self._task_info['CXPB'])

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register('min', np.min, axis=0)
        stats.register('max', np.max, axis=0)
        stats.register('mean', np.mean, axis=0)
        stats.register('std', np.std, axis=0)

        logbook = tools.Logbook()
        logbook.header = 'gen', 'evals', 'std', 'min', 'avg', 'max'

        pop = toolbox.population(n=pop_size)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        pop = toolbox.select(pop, len(pop))

        # Begin the generational process
        for gen in range(1, NGEN):

            # Vary the population
            offspring = tools.selTournamentDCD(pop, len(pop))
            offspring = [toolbox.clone(ind) for ind in offspring]

            for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                if random.random() <= CXPB:
                    toolbox.mate(ind1, ind2)

                toolbox.mutate(ind1)
                toolbox.mutate(ind2)
                del ind1.fitness.values, ind2.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Select the next generation population
            pop = toolbox.select(pop + offspring, pop_size)
            record = stats.compile(pop)

            # storage each generation result
            self._ga_results_list.append('%s\n' % (record['max'][0]))

            logbook.record(gen=gen, evals=len(invalid_ind), **record)
            if self._show_opts_history:
                print(logbook.stream)

        best_ind = tools.selBest(pop, 1)[0]
        best_material = self._fitness_obj.recovery_material(best_ind)
        return best_material
