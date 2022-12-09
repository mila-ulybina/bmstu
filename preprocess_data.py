__author__ = 'muscles.ai'

import os
import zipfile
import argparse
import numpy as np
import _pickle as cp

from io import BytesIO
from pandas import Series

#Число информационно-измерительных каналов
NB_SENSOR_CHANNELS = 113

#Файлы с измерительной информацией
OPPORTUNITY_DATA_FILES = ['dataset/P1_1.dat',
                          'dataset/P1_2.dat',
                          'dataset/P1_3.dat',
                          'dataset/P1_4.dat',
                          'dataset/P1_5.dat',
                          'dataset/P1_6.dat',
                          'dataset/P2_1.dat',
                          'dataset/P2_2.dat',
                          'dataset/P2_3.dat',
                          'dataset/P2_4.dat',
                          'dataset/P3_1.dat',
                          'dataset/P3_2.dat',
                          'dataset/P3_3.dat',
                          'dataset/P3_4.dat',
                          'dataset/P2_5.dat',
                          'dataset/P2_6.dat',
                          'dataset/P3_5.dat',
                          'dataset/P3_6.dat'
                          ]


#Пороговые значения для определения глобальных максимумов и минимумов для каждого из 113 каналов ввода
NORM_MAX_THRESHOLDS = [3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,
                       3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,
                       3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,
                       3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,
                       3000,   3000,   3000,   10000,  10000,  10000,  1500,   1500,   1500,
                       3000,   3000,   3000,   10000,  10000,  10000,  1500,   1500,   1500,
                       3000,   3000,   3000,   10000,  10000,  10000,  1500,   1500,   1500,
                       3000,   3000,   3000,   10000,  10000,  10000,  1500,   1500,   1500,
                       3000,   3000,   3000,   10000,  10000,  10000,  1500,   1500,   1500,
                       5000,   5000,   5000,    5000,   5000,   5000,  5000,   5000,   5000,
                       10000,  10000,  10000,  10000,  10000,  10000,  5000,   5000,   5000,
                       5000,    5000,   5000,   5000,   5000,   5000,   5000,   10000,  10000,
                       10000,  10000,  10000,  10000,  5000, ]

NORM_MIN_THRESHOLDS = [-3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,
                       -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,
                       -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,
                       -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,
                       -3000,  -3000,  -3000,  -10000, -10000, -10000, -1000,  -1000,  -1000,
                       -3000,  -3000,  -3000,  -10000, -10000, -10000, -1000,  -1000,  -1000,
                       -3000,  -3000,  -3000,  -10000, -10000, -10000, -1000,  -1000,  -1000,
                       -3000,  -3000,  -3000,  -10000, -10000, -10000, -1000,  -1000,  -1000,
                       -3000,  -3000,  -3000,  -10000, -10000, -10000, -1000,  -1000,  -1000,
                       -5000,  -5000,  -5000,  -5000,  -5000,  -5000,  -5000,  -5000,  -5000,
                       -10000, -10000, -10000, -10000, -10000, -10000, -5000,  -5000,  -5000,
                       -5000,  -5000,  -5000,  -5000,  -5000,  -5000,  -5000,  -10000, -10000,
                       -10000, -10000, -10000, -10000, -5000, ]


def select_columns_opp(data):
    """Выбор 113 столбцов данных

    Параметры:
        data: целочисленный numpy-массив измерительной информации (для всех каналов)

    Выход:
        Целочисленный numpy-массив выбранных каналов
    """

    #Исключение не используемых столбцов из данных
    features_delete = np.arange(46, 50)
    features_delete = np.concatenate([features_delete, np.arange(59, 63)])
    features_delete = np.concatenate([features_delete, np.arange(72, 76)])
    features_delete = np.concatenate([features_delete, np.arange(85, 89)])
    features_delete = np.concatenate([features_delete, np.arange(98, 102)])
    features_delete = np.concatenate([features_delete, np.arange(134, 243)])
    features_delete = np.concatenate([features_delete, np.arange(244, 249)])
    return np.delete(data, features_delete, 1)


def normalize(data, max_list, min_list):
    """Нормализация всех измерительных каналов

    Параметры:
        data: целочисленный numpy-массив измерительной информации
        max_list: целочисленный numpy-массив содержащий максимальные значения для каждого из 113 измерительных каналов
        min_list: целочисленный numpy-массив содержащий минимальные значения для каждого из 113 измерительных каналов

    Выход:
        Нормализованная измерительная информация
    """
    max_list, min_list = np.array(max_list), np.array(min_list)
    diffs = max_list - min_list
    for i in np.arange(data.shape[1]):
        data[:, i] = (data[:, i]-min_list[i])/diffs[i]
    #Проверка границ
    data[data > 1] = 0.99
    data[data < 0] = 0.00
    return data


def divide_x_y(data, label):
    """Сегментация каждого образца на признаки и метки

    Параметры:
        data: целочисленный numpy-массив измерительной информации
        label: строка, ['gestures' (default), 'locomotion'] вид деятельности подлежащий распознаванию

    Вывод:
        целочисленный numpy-массив,
        Признаки, инкапсулированные в матрицу и метки в виде массива
    """

    data_x = data[:, 1:114]
    if label not in ['locomotion', 'gestures']:
            raise RuntimeError("Ошибочная метка: '%s'" % label)
    if label == 'locomotion':
        data_y = data[:, 114]  #Метка локомоции
    elif label == 'gestures':
        data_y = data[:, 115]  #Метка жестов

    return data_x, data_y


def adjust_idx_labels(data_y, label):
    """Преобразование оригинальных меток в диапазон [0, nb_labels-1]

    Параметры:
        data_y: целочисленный numpy-массив меток датчиков
        label: строка, ['gestures' (default), 'locomotion'] вид деятельности, подлежащий распознаванию

    Выход:
        целочисленный numpy-массив
        Модифицированные метки датчиков
    """

    if label == 'locomotion':  #Метки для перемещений
        data_y[data_y == 4] = 3
        data_y[data_y == 5] = 4
    elif label == 'gestures':  #Метки для жестов
        data_y[data_y == 406516] = 1
        data_y[data_y == 406517] = 2
        data_y[data_y == 404516] = 3
        data_y[data_y == 404517] = 4
        data_y[data_y == 406520] = 5
        data_y[data_y == 404520] = 6
        data_y[data_y == 406505] = 7
        data_y[data_y == 404505] = 8
        data_y[data_y == 406519] = 9
        data_y[data_y == 404519] = 10
        data_y[data_y == 406511] = 11
        data_y[data_y == 404511] = 12
        data_y[data_y == 406508] = 13
        data_y[data_y == 404508] = 14
        data_y[data_y == 408512] = 15
        data_y[data_y == 407521] = 16
        data_y[data_y == 405506] = 17
    return data_y


def check_data(data_set):
    """Проверяет наличие файла с набором данных в папке с данными
       Если файл не найден, производится попытка его загрузки из источника

    Параметры:
       data_set: путь к оригинальному zip-файлу

    Выход:
    """
    print('Проверка датасета {0}'.format(data_set))
    data_dir, data_file = os.path.split(data_set)
    #Если каталог не указан, проверяем находится ли набор данных в каталоге data
    if data_dir == "" and not os.path.isfile(data_set):
        new_path = os.path.join(os.path.split(__file__)[0], "data", data_set)
        if os.path.isfile(new_path) or data_file == 'dataset.zip':
            data_set = new_path

    #Если набор данных не найден, попробуйте загрузить его из сетевого хранилища.
    if (not os.path.isfile(data_set)) and data_file == 'dataset.zip':
        print('... путь к набору данных {0} не найден'.format(data_set))
        import urllib
        origin = (
            ''
        )
        if not os.path.exists(data_dir):
            print('... создаем директорию {0}'.format(data_dir))
            os.makedirs(data_dir)
        print('... загрузка данных с {0}'.format(origin))
        urllib.request.urlretrieve(origin, data_set)

    return data_dir


def process_dataset_file(data, label):
    """Конвеер для обработки отдельных файлов измерительной информации

    Параметры:
        data: целочисленная numpy-матрица
        Матрица, содержащая образцы данных (строки) для каждого измерительного канала (столбца)
        label: string, ['gestures' (default), 'locomotion'] - тип деятельности, которая будет распознаваться

    Выход:
        целочисленная numpy-матрица, целочисленный numpy-массив
        Обработанная измерительная информация, разделенная на признаки (x) и метки (y)
    """

    #Выбор нужного столбца
    data = select_columns_opp(data)

    #Разделение столбцов на признаки и метки
    data_x, data_y =  divide_x_y(data, label)
    data_y = adjust_idx_labels(data_y, label)
    data_y = data_y.astype(int)

    #Выполнение линейной интерполяции
    data_x = np.array([Series(i).interpolate() for i in data_x.T]).T

    #Недостающие данные заменяем нулями
    data_x[np.isnan(data_x)] = 0

    #Нормализация всех измерительных каналов
    data_x = normalize(data_x, NORM_MAX_THRESHOLDS, NORM_MIN_THRESHOLDS)

    return data_x, data_y


def generate_data(dataset, target_filename, label):
    """Чтение сырых данных и обработка всех измерительных каналов

    Параметры:
        dataset: строка - путь к zip-файлу
        target_filename: строка - обрабатываемый файл
        label: строка - ['gestures' (default), 'locomotion']
            Вид деятельности, подлежащий распознаванию.
    """

    data_dir = check_data(dataset)

    data_x = np.empty((0, NB_SENSOR_CHANNELS))
    data_y = np.empty((0))

    zf = zipfile.ZipFile(dataset)
    print('Обработка файлов с набором данных ...')
    for filename in OPPORTUNITY_DATA_FILES:
        try:
            data = np.loadtxt(BytesIO(zf.read(filename)))
            print('... файл {0}'.format(filename))
            x, y = process_dataset_file(data, label)
            data_x = np.vstack((data_x, x))
            data_y = np.concatenate([data_y, y])
        except KeyError:
            print('ОШИБКА: Не найден {0} в zip-файле'.format(filename))

    nb_training_samples = 557963
    X_train, y_train = data_x[:nb_training_samples,:], data_y[:nb_training_samples]
    X_test, y_test = data_x[nb_training_samples:,:], data_y[nb_training_samples:]

    print("Размер итогового датасета: | train {0} | test {1} | ".format(X_train.shape,X_test.shape))

    obj = [(X_train, y_train), (X_test, y_test)]
    f = open(os.path.join(data_dir, target_filename), 'wb')
    cp.dump(obj, f, protocol=-1)
    f.close()


def get_args():
    '''Функция анализа аргументов командной строки'''
    parser = argparse.ArgumentParser(
        description='Препроцессинг данных')
    # Анализ аргументов
    parser.add_argument(
        '-i', '--input', type=str, help='Zip-файл с данными', required=True)
    parser.add_argument(
        '-o', '--output', type=str, help='Обработка фала с данными', required=True)
    parser.add_argument(
        '-t', '--task', type=str.lower, help='Тип обработки', default="gestures", choices = ["gestures", "locomotion"], required=False)
    #Массив всех аргументов, переданных скрипту
    args = parser.parse_args()
    #Назначение args переменных
    dataset = args.input
    target_filename = args.output
    label = args.task
    #Возвращаем все значения переменных
    return dataset, target_filename, label

if __name__ == '__main__':

    Dataset_zip, output, l = get_args()
    generate_data(Dataset_zip, output, l)
