import tkinter as tk
import time
import csv

status = False
l = 0
r = 0

csv_file_path = "intervale_rotiri_spate_calin.csv"

def millis():
    now = time.time()  
    midnight = time.mktime(time.strptime(time.strftime("%Y-%m-%d"), "%Y-%m-%d")) 
    milliseconds = (now - midnight) * 1000  
    return int(milliseconds)

def button_click():
    global status, l, r
    print("Button clicked")
    if status == False:
        l = millis()
    else:
        r = millis()
        csv_writer.writerow([l, r])
        print(f"Interval: {r - l} ms")
    status = not status

def simulate_button_press(event):
    button_click()

with open(csv_file_path, mode='a', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    
    root = tk.Tk()
    root.title("Simulate Button Press")

    button = tk.Button(root, text="Click Me", command=button_click)
    button.pack()

    root.bind('<Return>', simulate_button_press)

    root.mainloop()