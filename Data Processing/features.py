import csv
import math

filePath = "data/data_set3.csv"
outFile = "data/features_data_set3.csv"
nrFeatures = 24
sizeFeature = 500
alpha = float(0)

fileAns = open(outFile, mode='a', newline='')
writer = csv.writer(fileAns)

with open(filePath, 'r') as file:
    csv_reader = csv.reader(file)
    indd = 0

    test = 0
    for row in csv_reader:
        test += 1
        if test % 1000 == 0:
            print(test)

        i2 = 0
        ans = []
        while i2 < nrFeatures:
            i = i2 * sizeFeature
            MAV = 0
            ZCR = 0
            WL = 0
            SSC = 0
            RMS = 0
            ISEMG = 0
            while i < (i2 + 1) * sizeFeature:
                MAV += abs(float(row[i]))
                RMS += (float(row[i]) * float(row[i]))
                ISEMG += math.sqrt(abs(float(row[i])))
                if i != i2 * sizeFeature:
                    if ( float(row[i]) * float(row[i - 1]) <= 0 and abs(float(row[i]) - float(row[i - 1])) >= alpha ):
                        ZCR += 1
                    WL += abs(float(row[i]) - float(row[i - 1]))
                    if i < (i2 + 1) * sizeFeature:
                        if ((float(row[i]) - float(row[i - 1])) * (float(row[i]) - float(row[i + 1]))) >= alpha:
                            SSC += 1
                i += 1

            RMS /= float(sizeFeature)
            RMS = math.sqrt(RMS)
            MAV /= float(sizeFeature)

            Hjorth = 0
            i = i2 * sizeFeature
            while i < (i2 + 1) * sizeFeature:
                Hjorth += ((float(row[i]) - MAV) * (float(row[i]) - MAV))
                i += 1
            Hjorth /= float(sizeFeature)
            Hjorth = math.sqrt(Hjorth)

            Skew = 0
            i = i2 * sizeFeature
            while i < (i2 + 1) * sizeFeature:
                Skew += (((float(row[i]) - MAV) / Hjorth) * ((float(row[i]) - MAV) / Hjorth) * ((float(row[i]) - MAV) / Hjorth))
                i += 1

            ans.append(MAV)
            ans.append(ZCR)
            ans.append(WL)
            ans.append(SSC)
            ans.append(RMS)
            ans.append(Hjorth)
            ans.append(Skew)
            ans.append(ISEMG)
            i2 += 1
        ans.append(float(row[len(row) - 2]))
        writer.writerow(ans)
        #print("is ok")


fileAns.close()