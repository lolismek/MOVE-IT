import extract
import csv

names = ["alex", "andrei", "calin", "costea", "emi", "guzu", "iulian", "mariana", "raresf", "rares", "teo", "toma", "subiect1", "subiect2", "subiect3", "subiect4", "subiect5", "subiect6", "subiect7", "subiect8", "subiect9", "subiect10", "subiect11"]
exercises = ["alergare", "mers", "jumping_jack", "sarituri", "rotire_fata", "rotire_spate"]


csv_file_path = "data/data_set3.csv"

with open(csv_file_path, mode='a', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)

    for i in range(0, 23):
        for j in range(0, 6):
            try: 
                #data = extract.extract_samples(exercises[j] + "_" + names[i], 2, 500, 0.5, 2, j, i, 0)
                data = extract.extract_samples(exercises[j] + "_" + names[i], 4, 500, 0.2, 1, j, i, 0)
                for row in data:
                    csv_writer.writerow(row)
                print("<< " + str(i) + " " + str(j))
            except Exception as e:
                print(">> " + str(i) + " " + str(j))
                print(e)