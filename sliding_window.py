__author__ = 'muscles.ai'

import numpy as np
from numpy.lib.stride_tricks import as_strided as ast

def norm_shape(shape):
    '''
    Нормализация массива numpy.

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

def sliding_window(a, ws, ss=None, flatten=True):
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
    ws = norm_shape(ws)
    ss = norm_shape(ss)

    #Преобразование ws, ss и a.shape в numpy-массивы, для работы сразу во всех измерениях
    ws = np.array(ws)
    ss = np.array(ss)
    shape = np.array(a.shape)


    #Проверяем, что ws, ss, и a.shape имеют одинаковое число измерений
    ls = [len(shape), len(ws), len(ss)]
    if 1 != len(set(ls)):
        raise ValueError('a.shape, ws и ss должны иметь одинаковую длину. Сейчас они %s' % str(ls))

    #Проверяем, что ws меньше, чем каждое из измерений
    if np.any(ws > shape):
        raise ValueError('ws не может быть больше любого измерения. a.shape равен %s и ws равен %s' % (str(a.shape), str(ws)))

    #Вычисляем число срезов в каждом измерении
    newshape = norm_shape(((shape - ws) // ss) + 1)
    #Форма массива будет равна количеству срезов в каждом измерении плюс форма окна (добавление кортежа)
    newshape += norm_shape(ws)
    newstrides = norm_shape(np.array(a.strides) * ss) + a.strides
    strided = ast(a, shape=newshape, strides=newstrides)
    if not flatten:
        return strided

    meat = len(ws) if ws.shape else 0
    firstdim = (np.product(newshape[:-meat]),) if ws.shape else ()
    dim = firstdim + (newshape[-meat:])
    return strided.reshape(dim)