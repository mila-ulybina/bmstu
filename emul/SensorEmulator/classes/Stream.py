import numpy as np
import pandas as pd
import os 
import zipfile
from io import BytesIO
import random
import math
import itertools

class Stream:
    NB_SENSOR_CHANNELS = 113

    OPPORTUNITY_DATA_FILES = [
                          'dataset/P2_5.dat',
                          'dataset/P2_6.dat',
                          'dataset/P3_5.dat',
                          'dataset/P3_6.dat'
                          ]
    activity_dict = {
        'Ходьба на месте': [1, 2.5, 7],
        'Приседание': [1, 2.5, 7],
        'Жим ногами': [1, 3, 6],
        'Мах левой ногой': [1, 3, 6],
        'Мах правой ногой': [1, 2, 8],
        'Классическое отжимание': [0.5, 4, 10],
        'Узкие отжимания': [1, 3.5, 5.5],
        'Мах левой рукой': [1, 3.5, 6],
        'Мах правой рукой': [1, 2, 3],
        'Езда на велотренажере': [0.5, 2, 5],
        'Прыжок': [1, 2, 4],
        'Вращение руками': [0.5, 1.5, 5],
        'Вращение в локтях': [1.5, 2.5, 4],
        'Выпад': [0.5, 2.5, 10],
        'Наклон корпуса влево': [1, 5, 20],
        'Наклон корпуса вправо': [1, 10, 50],
        'toggle_switch': [0.5, 1.5, 10]
        } 

    


    def __init__(self, static_path):
        
        self.static_path = static_path
        print("Loading data...")
        X_test, y_test = self.generate_data(os.path.join(self.static_path,'dataset.zip'), 'gestures')
        self.test_df = self.define_test_df(X_test, y_test)

    def get_data(self):
        return self.test_df

    def get_data_lbl(self, lbl):

        activiti_df = self.test_df[self.test_df.lbl == lbl].reset_index(drop=True) 
        activiti_df = activiti_df.iloc[:,1:115]
        activiti_np = activiti_df.to_numpy()
        return activiti_np


    def stream_by_lbl2(self, dataset, index, batch_size = 24):

        return self.circ_slice(dataset, index*batch_size, batch_size)


    def stream_by_lbl(self, lbl, dataset, batch_size = 24):

        activiti_df = dataset[dataset.lbl == lbl].reset_index(drop=True) 
        activiti_df = activiti_df.iloc[:,1:115]
        activiti_np = activiti_df.to_numpy()
        n_samples = math.floor(len(activiti_df)/batch_size)
        indices = np.arange(n_samples)
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)

            batch_idx = indices[start:end]

            yield activiti_np[batch_idx]

    def start_streaming(self):
        while True:
            self.null_class_stream()
            self.random_stream()

    def null_class_stream(self):

        print('Нулевая активность')
        activity_label = 0
        print('Метка нулевой активности', activity_label)
        streaming_time = round(random.triangular(2, 5, 10),1)
        print('Случайное определенние тайминга нулевого потока', streaming_time)
        sliced = self.sample_slice(activity_label, streaming_time)
        return sliced

    def random_stream(self):

        random_activity = random.choice(list(self.activity_dict))
        print('Случайная активность', random_activity)
        activity_label = list(self.activity_dict.keys()).index(random_activity)+1
        print('Метка случайной активности', activity_label)
        streaming_time = round(random.triangular(self.activity_dict[random_activity][0], self.activity_dict[random_activity][2], self.activity_dict[random_activity][1]),1)
        print('Случайно определенный тайминг потока', streaming_time)

        sliced = self.sample_slice(activity_label, streaming_time)
        #Добавлять после каждого цикла тайминга немного из цикла нулевого класса
        return sliced

    def sample_slice(self, activity_label, streaming_time):

        activiti_df = self.test_df[self.test_df.lbl == activity_label].reset_index(drop=True)
        activiti_df_indexes= list(activiti_df.index)
        rand_start_pos = random.choice(activiti_df_indexes)
        samples =  math.floor(streaming_time/0.033)
        sliced = self.circ_slice(list(activiti_df.index), rand_start_pos, samples)

        print('Сколько надо образцов',streaming_time/0.033, 'округление',samples)
        print('Сколько есть образцов',len(activiti_df))
        print('Индекс случайного старт-образца',rand_start_pos)
        print(sliced)
        print('Итого образцов в случайном тайминге',len(sliced))
        return sliced


    def circ_slice(self, a, start, length):
        it = itertools.cycle(a)
        next(itertools.islice(it, start, start), None)
        return np.array(list(itertools.islice(it, length)))

    def define_test_df(self, X_test, y_test):
        test_df = pd.DataFrame(X_test)
        test_df['lbl'] = y_test
        return test_df


    def generate_data(self, dataset, label):
        """Функция чтения сырых данных и обработка всех измерительных каналов

        Параметры:
            dataset: строка - путь к  zip-файлу
            target_filename: строка - обрабатываемый файл
            label: строка - ['gestures' (default), 'locomotion']
                Вид деятельности, подлежащий распознаванию.
        """
        data_dir = self.check_data(dataset)

        data_x = np.empty((0, self.NB_SENSOR_CHANNELS+1))
        data_y = np.empty((0))

        zf = zipfile.ZipFile(dataset)
        print('Обработка файлов с набором данных ...')
        for filename in self.OPPORTUNITY_DATA_FILES:
            try:
                data = np.loadtxt(BytesIO(zf.read(filename)))
                print('... файл {0}'.format(filename))
                #Выбор нужного столбца
                data = self.select_columns_opp(data)
                #Разделение столбцов на признаки и метки
                x, y =  self.divide_x_y(data, label)
                y = self.adjust_idx_labels(y, label)
                y = y.astype(int)
                data_x = np.vstack((data_x, x))
                data_y = np.concatenate([data_y, y])
            except KeyError:
                print('ОШИБКА: Не найден {0} в zip-файле'.format(filename))

        X_test, y_test = data_x, data_y

        print("Итоговый размер датасета: | train NO | test {0} | ".format(X_test.shape))

        return X_test, y_test


    def divide_x_y(self, data, label):
        """Сегментирует каждый образец на признаки и метки

        Параметры:
            data: целочисленный numpy-массив измерительной информации
            label: строка, ['gestures' (default), 'locomotion'] вид деятельности подлежащий распознаванию

        Вывод:
            целочисленный numpy-массив,
            Признаки, инкапсулированные в матрицу и метки в виде массива
        """

        data_x = data[:, 0:114] #не забыть вернуть 114!
        if label not in ['locomotion', 'gestures']:
                raise RuntimeError("Invalid label: '%s'" % label)
        if label == 'locomotion':
            data_y = data[:, 114]  #Метка локомоции
        elif label == 'gestures':
            data_y = data[:, 115]  #Метка жестов

        return data_x, data_y


    def select_columns_opp(self, data):
        """Выбор 113 столбцов данных

        Параметры:
            data: целочисленный numpy-массив измерительной информации

        Выход:
            Целочисленный numpy-массив выбранных каналов
        """

        #исключение данных
        features_delete = np.arange(46, 50)
        features_delete = np.concatenate([features_delete, np.arange(59, 63)])
        features_delete = np.concatenate([features_delete, np.arange(72, 76)])
        features_delete = np.concatenate([features_delete, np.arange(85, 89)])
        features_delete = np.concatenate([features_delete, np.arange(98, 102)])
        features_delete = np.concatenate([features_delete, np.arange(134, 243)])
        features_delete = np.concatenate([features_delete, np.arange(244, 249)])
        return np.delete(data, features_delete, 1)

    def check_data(self, data_set):
        """Проверяет наличие файла с набором данных в папке с данными
        Если файл не найден, производится попытка его загрузки из источника

        Параметры:
        data_set: путь к zip-файлу

        Выход:
        """
        print('Checking dataset {0}'.format(data_set))
        data_dir, data_file = os.path.split(data_set)
        # Если каталог не указан, проверяем находится ли набор данных в каталоге data
        if data_dir == "" and not os.path.isfile(data_set):
            new_path = os.path.join(os.path.split(__file__)[0], "data", data_set)
            if os.path.isfile(new_path) or data_file == 'dataset.zip':
                data_set = new_path

        # Если набор данных не найден, попробуйте загрузить его из сетевого хранилища.
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

    def adjust_idx_labels(self, data_y, label):
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