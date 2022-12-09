import os
import numpy as np
import socket
import struct
from classes.Preprocess import Preprocess
from classes.Model import Model
import tkinter
import threading
import queue

activity_dict = [
        'Ходьба на месте',
        'Приседание',
        'Жим ногами',
        'Мах левой ногой',
        'Мах правой ногой',
        'Классическое отжимание',
        'Узкие отжимания',
        'Мах левой рукой',
        'Мах правой рукой',
        'Езда на велотренажере',
        'Прыжок',
        'Вращение руками',
        'Вращение в локтях',
        'Выпад',
        'Наклон корпуса влево',
        'Наклон корпуса вправо',
        'Нулевой класс'
        ]

host = os.getenv('CLASSIFIER_HOST', 'localhost')    #host
HTTP_SERVER_PORT = 5001      #port

base_dir = os.getcwd()
static_path = os.path.join(base_dir, 'static')
model_name = 'convLstm_noNULL.h5'

preprocess = Preprocess()
model = Model(static_path, model_name)

HOST = '127.0.0.1'
PORT = 5001

def recv_msg(sock):
    #Чтение длины сообщения и распаковка его в целое число
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    #Чтение данных сообщения
    return recvall(sock, msglen)

def recvall(sock, n):
    #Вспомогательная функция для приема n байт или возвращает None при достижении EOF
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data



class GuiPart:
    def __init__(self, master, queue, endCommand):
        self.queue = queue
        #Настройка GUI
        console = tkinter.Button(master, text='Finish', command=endCommand)
        console.config(font=("Courier", 20))
        console.place(relx=.36, rely=.8)
        # Add more GUI stuff here depending on your specific needs
        l0 = tkinter.Label(text='Label of the motion:')
        l0.config(font=("Courier", 32))
        l0.place(relx=.05, rely=.35)
        self.var = tkinter.StringVar()
        self.var.set('Hello')
        l = tkinter.Label(master, textvariable = self.var)
        l.config(font=("Courier", 44))
        l.place(relx=.1, rely=.6)
        l1 = tkinter.Label(text='Num of the motion:')
        l1.config(font=("Courier", 24))
        l1.place(relx=.6, rely=.06)
        self.var_count = tkinter.IntVar()
        self.var_count.set(0)
        l_count = tkinter.Label(master, textvariable = self.var_count, fg="#eee", bg="#333")
        l_count.config(font=("Courier", 32))
        l_count.place(relx=.9, rely=.05)

    def processIncoming(self):
        """Обрабатывать все сообщения в очереди, если таковые имеются"""
        while self.queue.qsize(  ):
            try:
                msg = self.queue.get(0)
                self.var.set(str(msg[0]))
                self.var_count.set(msg[1])

            except queue.Empty:
                pass

class ThreadedClient:
    """
    Запуск GUI и рабочего потока
    """
    def __init__(self, master):
        """
        Запуск графического интерфейса и асинхронных потоков.
        Создание новго потока для рабочего процесса (I/O).
        """
        self.counter = 0
        self.master = master
        self.queue = queue.Queue(  )

        self.gui = GuiPart(master, self.queue, self.endApplication)

        #Настройка потока для выполнения асинхронного ввода-вывода
        self.running = 1
        self.thread1 = threading.Thread(target=self.workerThread1)
        self.thread1.start(  )

        #Проверка очереди
        self.periodicCall(  )

    def periodicCall(self):
        """
        Проверяем каждые 200 мс, есть ли что-то новое в очереди.
        """
        self.gui.processIncoming(  )
        if not self.running:
            import sys
            sys.exit(1)
        self.master.after(200, self.periodicCall)

    def workerThread1(self):
        """
        Обрабатка асинхронного ввода-вывода.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            while self.running:
                request_message = 'Готов принять данные'
                request_message = request_message.encode('utf-8')
                s.sendall(request_message)
                # data = s.recv(4096) 
                data = recv_msg(s)
                if not data: break
                batch_numpy = np.frombuffer(data, dtype=float)
                
                batch_ready = preprocess.preprocess_batch(batch_numpy)
                label_streaming = model.predict(batch_ready)
                out = activity_dict[label_streaming[0]-1]

                self.counter += 1
                self.queue.put((out,self.counter))

            s.close()

    def endApplication(self):
        self.running = 0


root = tkinter.Tk()
root.title("Sensor emulator")
root.minsize(300, 300)

client = ThreadedClient(root)
root.mainloop()