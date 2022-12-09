import os
import socket
import threading
import select
from classes.Stream import Stream
import struct
import math


base_dir = os.getcwd()
static_path = os.path.join(base_dir, 'static')
stream = Stream(static_path)

activity_dict = [
        'open_door1',
        'open_door2',
        'close_door1',
        'close_door2',
        'open_fridge',
        'close_fridge',
        'open_dishwasher',
        'close_dishwasher',
        'open_drawer1',
        'close_drawer1',
        'open_drawer2',
        'close_drawer2',
        'open_drawer3',
        'close_drawer3',
        'clean_table',
        'drink_cup',
        'toggle_switch'
        ]


HOST = '127.0.0.1'
PORT = 5001

def send_msg(sock, msg):
    msg = struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)

def socket_thread():
    stop = False
    while not stop:
        global client_sock
        global client_addr
        global stop_threads
        global label
        if stop_threads:
            print('Закрытие соединения сервера {}'.format(client_addr))
            client_sock.close()
            break
        if client_sock:
            #Проверяем, подключен ли еще клиент и доступны ли данные:
            try:
                rdy_read, rdy_write, sock_err = select.select([client_sock,], [], [])
            except select.error:
                print('Ошибка сокета {}'.format(client_addr))
                return 1

            if len(rdy_read) > 0:
                read_data = client_sock.recv(255)
                #Проверяем, закрыт ли сокет
                if len(read_data) == 0:
                    print('{} закрыт сокет.'.format(client_addr))
                    stop = True
                else:
                    print('>>> Полученный: {}'.format(read_data.rstrip()))
                    if read_data.rstrip() == 'quit':
                        stop = True
                    else:

                        message  = 'Поток с меткой '+ str(label)
                        message = message.encode('utf-8')
                        client_sock.send(message)
                        # portion_num += 1
        else:
            print("Клиент не подключен, SocketServer не может получать данные")
            stop = True

        # Close socket
        print('Закрытие связи с {}'.format(client_addr))
        client_sock.close()
        return 0

def socket_thread2():
    stop = False
    try:
        while not stop:
            global client_sock
            global client_addr
            global stop_threads
            global label
            global lbl_sensor_data
            global index
            global n_samples
            if stop_threads:
                print('Закрытие соединения от сервера {}'.format(client_addr))
                client_sock.close()
                break
            data = client_sock.recv(1024)
            data = data.decode('utf-8')
            if not data:
                print('Нет запроса от клиента... Ждите!')

            if index > n_samples:
                index = len(lbl_sensor_data) % index
            batch_numpy = stream.stream_by_lbl2(lbl_sensor_data, index)
            index += 1

            batch_numpy = batch_numpy.tobytes()
            send_msg(client_sock, batch_numpy)
    except StopIteration:
        print('except StopIteration')


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    print('binded')
    s.listen()
    print('listened')
    client_sock, client_addr = s.accept()
    # with client_sock:
    print('Connected by', client_addr)
    label = 0
    index = 0
    batch_size = 24
    input_lbl = ''
    stop_threads = False
    lbl_sensor_data = stream.get_data_lbl(label)
    n_samples = math.floor(len(lbl_sensor_data)/batch_size)
    t1 = threading.Thread(target=socket_thread2)
    t1.start()
    
    while input_lbl != 'q':
        input_lbl = input()
        if input_lbl != '':
            print(input_lbl)
            label_int = int(input_lbl)
            if label_int in range(0, 18):                
                label = label_int
                print(activity_dict[label-1])
                lbl_sensor_data = stream.get_data_lbl(label)
                n_samples = math.floor(len(lbl_sensor_data)/batch_size)
                index = 0

    stop_threads = True
    t1.join()
    print('finish')
