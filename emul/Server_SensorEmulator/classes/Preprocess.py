import os 
import numpy as np
import pandas as pd

from numpy.lib.stride_tricks import as_strided as ast

class Preprocess(object):

    NB_SENSOR_CHANNELS = 113

    SLIDING_WINDOW_LENGTH = 24

    SLIDING_WINDOW_STEP = 12

    NORM_MAX_THRESHOLDS = [3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,
                        3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,
                        3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,
                        3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,
                        3000,   3000,   3000,   10000,  10000,  10000,  1500,   1500,   1500,
                        3000,   3000,   3000,   10000,  10000,  10000,  1500,   1500,   1500,
                        3000,   3000,   3000,   10000,  10000,  10000,  1500,   1500,   1500,
                        3000,   3000,   3000,   10000,  10000,  10000,  1500,   1500,   1500,
                        3000,   3000,   3000,   10000,  10000,  10000,  1500,   1500,   1500,
                        250,    25,     200,    5000,   5000,   5000,   5000,   5000,   5000,
                        10000,  10000,  10000,  10000,  10000,  10000,  250,    250,    25,
                        200,    5000,   5000,   5000,   5000,   5000,   5000,   10000,  10000,
                        10000,  10000,  10000,  10000,  250, ]

    NORM_MIN_THRESHOLDS = [-3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,
                        -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,
                        -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,
                        -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,
                        -3000,  -3000,  -3000,  -10000, -10000, -10000, -1000,  -1000,  -1000,
                        -3000,  -3000,  -3000,  -10000, -10000, -10000, -1000,  -1000,  -1000,
                        -3000,  -3000,  -3000,  -10000, -10000, -10000, -1000,  -1000,  -1000,
                        -3000,  -3000,  -3000,  -10000, -10000, -10000, -1000,  -1000,  -1000,
                        -3000,  -3000,  -3000,  -10000, -10000, -10000, -1000,  -1000,  -1000,
                        -250,   -100,   -200,   -5000,  -5000,  -5000,  -5000,  -5000,  -5000,
                        -10000, -10000, -10000, -10000, -10000, -10000, -250,   -250,   -100,
                        -200,   -5000,  -5000,  -5000,  -5000,  -5000,  -5000,  -10000, -10000,
                        -10000, -10000, -10000, -10000, -250, ]


    def preprocess_batch(self, batch):
        batch = batch.reshape((-1, self.NB_SENSOR_CHANNELS+1))

        #Убираем последний lbl-столбец
        batch = batch[:,:-1]

        batch = np.array([pd.Series(i).interpolate() for i in batch.T]).T
        batch[np.isnan(batch)] = 0  # оставшиеся nan на 0-вом индексе после интерполяции

        # Нормализация всех измерительных каналов
        batch = self.normalize(batch, self.NORM_MAX_THRESHOLDS, self.NORM_MIN_THRESHOLDS)

        batch = self.opp_sliding_window(batch, self.SLIDING_WINDOW_LENGTH, self.SLIDING_WINDOW_STEP)
        batch = batch.reshape((-1, self.NB_SENSOR_CHANNELS, self.SLIDING_WINDOW_LENGTH, 1))

        return batch

    def normalize(self, data, max_list, min_list):

        max_list, min_list = np.array(max_list), np.array(min_list)
        diffs = max_list - min_list
        for i in np.arange(data.shape[1]):
            data[:, i] = (data[:, i]-min_list[i])/diffs[i]

        data[data > 1] = 0.99
        data[data < 0] = 0.00
        return data
    
    def opp_sliding_window(self, data_x, ws, ss):
        assert self.NB_SENSOR_CHANNELS == data_x.shape[1]
        data_x = self.sliding_window(data_x,(ws,data_x.shape[1]),(ss,1))
        return data_x.astype(np.float32)

    def norm_shape(self, shape):
        '''
        Нормализация формы массива numpy, чтобы он всегда представлялся как кортеж, даже для одномерных фигур.

        Параметры:
            shape - int, или кортеж из int

        Выход:
            форма кортежа
        '''
        try:
            i = int(shape)
            return (i,)
        except TypeError:
            #Входной параметр не число
            pass

        try:
            t = tuple(shape)
            return t
        except TypeError:
            #Форма не может быть повторена
            pass

        raise TypeError('форма должна быть int, или кортеж из int')

    def sliding_window(self, a,ws,ss = None,flatten = True):
        '''
        Возвращает скользящее окно для любого числа измерений

        Параметры:
            a  - n-мерный массив numpy.
            ws - int (для 1D) или кортеж (для 2D или большего числа измерения) представления размера для каждого измерения окна.
            ss - int (для 1D) или кортеж (для 2D или большего числа измерений) представляет количество скользящих
                    окон для каждого измерения. Если не указано, по умолчанию используется ws.
            flatten - если True, все срезы сглаживаются, в противном случае для каждого измерения входных данных существует
                    дополнительное измерение.

        Выход:
            массив, содержащий каждое n-мерное окно из a
        '''

        if None is ss:
            #ss не определено. Окна не будут перекрываться ни в каком направлении.
            ss = ws
        ws = self.norm_shape(ws)
        ss = self.norm_shape(ss)

        #Преобразование ws, ss и a.shape в numpy-массивы, для работы сразу во всех измерениях
        # dimension at once.
        ws = np.array(ws)
        ss = np.array(ss)
        shape = np.array(a.shape)

        #Проверяем, что ws, ss, и a.shape имеют одинаковое число измерений
        ls = [len(shape),len(ws),len(ss)]
        if 1 != len(set(ls)):
            raise ValueError('a.shape, ws и ss должны иметь одинаковую длину. Сейчас они %s' % str(ls))

        # проверяем, что ws меньше, чем каждое из измерений
        if np.any(ws > shape):
            raise ValueError('ws не может быть больше любого измерения. a.shape равен %s и ws равен %s' % (str(a.shape),str(ws)))

        # Вычисляем число срезов в каждом измерении
        newshape = self.norm_shape(((shape - ws) // ss) + 1)
        newshape += self.norm_shape(ws)
        newstrides = self.norm_shape(np.array(a.strides) * ss) + a.strides
        strided = ast(a,shape = newshape,strides = newstrides)
        if not flatten:
            return strided

        meat = len(ws) if ws.shape else 0
        firstdim = (np.product(newshape[:-meat]),) if ws.shape else ()
        dim = firstdim + (newshape[-meat:])

        return strided.reshape(dim)