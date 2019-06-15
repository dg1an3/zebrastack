import os
import matplotlib.pyplot as plt

sshot_path = 'c:\\Users\\dglan\\OneDrive\\Pictures\\Screenshots_Timesnapper_Laptop\\'
for path, _, files in os.walk(sshot_path):
    maybe_date = path.split('\\')[-1].split('-')
    if len(maybe_date) < 3:
        continue
    date = [int(part) for part in maybe_date]
    for file in files:
        if not(file.endswith('png')):
            continue
        time = [int(part) for part in file.split('.')[0:3]]
        print(date[0], date[1], date[2], time[0], time[1], time[2])
