import socket
import time
import csv

#de verificat ca time-ul bratarilor are sens cand se da boot
#!!! intre intervale si senzori exista o eroare de 0.1s

import keyboard

def millis():
    now = time.time()  
    midnight = time.mktime(time.strptime(time.strftime("%Y-%m-%d"), "%Y-%m-%d")) 
    milliseconds = (now - midnight) * 1000  
    return int(milliseconds)

UDP_IP = "0.0.0.0"  
UDP_PORT = 3333

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print("UDP server started")

csv_file_path = "rotiri_spate_calin.csv"

with open(csv_file_path, mode='a', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)

    while True:
        data, addr = sock.recvfrom(1000000000) #bytes
        s = data.decode()
  
        s_vec = s.split(';')
        for el in s_vec:
            s_vec_vec = el.split(',')
            vals = [float(x) for x in s_vec_vec]
            print(vals)
            csv_writer.writerow(vals)

        #s_vec = s.split(';')

        #print(len(s_vec))
        #for el in s_vec:
            #s_vec_vec = el.split(',')
            #print(s_vec_vec)
            #vals = [float(x) for x in s_vec_vec]
            #print(vals)
            #csv_writer.writerow(vals)
        #s_vec = s.split(',')
        #vals = [float(x) for x in s_vec]

        #print(vals)
        #csv_writer.writerow(vals)

#notes:
# gyroscope outputs degrees / s
# acc outputs m/s^2
# ipconfig getifaddr en0
