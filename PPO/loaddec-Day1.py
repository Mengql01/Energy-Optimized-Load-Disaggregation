import time
import csv
import numpy as np
import math

def read__csvload():
    dataset = list()
    with open('load-day1-56,058.csv', 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
        for col in range(len(dataset[0])):
            for row in dataset:
                row[col] = float(row[col].strip())
    return dataset

def read__csv1():
    dataset = list()
    with open('dd.csv', 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
        for col in range(len(dataset[0])):
            for row in dataset:
                row[col] = float(row[col].strip())
    return dataset

def write_csv1(i6):
    path = "bb.csv"
    with open("bb.csv", 'w', encoding='utf-8', newline='') as f3:
        csv_write = csv.writer(f3)
        data_row = [i6]
        csv_write.writerow(data_row)

def write_csv2(i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13):
    path = "aa.csv"
    with open("aa.csv", 'w', encoding='utf-8', newline='') as f4:
        csv_write = csv.writer(f4)
        data_row = [i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13]
        csv_write.writerow(data_row)

def read__csv2():
    dataset = list()
    with open('cc.csv', 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
        for col in range(len(dataset[0])):
            for row in dataset:
                row[col] = float(row[col].strip())
    return dataset



read__csvload=read__csvload()
nStep = 1
Qs_sum = 0
list_state3_Qp = []
def main():
    global nStep
    global Qs_sum
    global list_state3_Qp

    Qb1 = read__csvload[0][1]
    Qb2 = read__csvload[1][1]
    Qb3 = read__csvload[2][1]
    Qb4 = read__csvload[3][1]
    Qb5 = read__csvload[4][1]
    Qb6 = read__csvload[5][1]
    Qb7 = read__csvload[6][1]
    Qb8 = read__csvload[7][1]
    Qb9 = read__csvload[8][1]
    Qb10 = read__csvload[9][1]
    Qb11 = read__csvload[10][1]
    Qb12 = read__csvload[11][1]

    state1_Qp01 = Qb1*(1-read__csv2()[0][0])
    state2_Qp02 = Qb2 * (1 - read__csv2()[0][1])
    state3_Qp03 = Qb3 * (1 - read__csv2()[0][2])
    state4_Qp04 = Qb4 * (1 - read__csv2()[0][3])
    state5_Qp05 = Qb5 * (1 - read__csv2()[0][4])
    state6_Qp06 = Qb6 * (1 - read__csv2()[0][5])
    state7_Qp07 = Qb7 * (1 - read__csv2()[0][6])
    state8_Qp08 = Qb8 * (1 - read__csv2()[0][7])
    state9_Qp09 = Qb9 * (1 - read__csv2()[0][8])
    state10_Qp010 = Qb10 * (1 - read__csv2()[0][9])
    state11_Qp011 = Qb11 * (1 - read__csv2()[0][10])
    state12_Qp012 = Qb12 * (1 - read__csv2()[0][11])
    Qs_sum=(Qb1-state1_Qp01)+(Qb2-state2_Qp02)+(Qb3-state3_Qp03)+(Qb4-state4_Qp04)+(Qb5-state5_Qp05)+(Qb6-state6_Qp06)+(Qb7-state7_Qp07)+(Qb8-state8_Qp08)+(Qb9-state9_Qp09)+(Qb10-state10_Qp010)+(Qb11-state11_Qp011)+(Qb12-state12_Qp012)
    Ssi =5605.808 - Qs_sum
    Ssi12 = Ssi/12
    state1_Qp1 = state1_Qp01-Ssi12
    state2_Qp2 = state2_Qp02-Ssi12
    state3_Qp3 = state3_Qp03-Ssi12
    state4_Qp4 = state4_Qp04-Ssi12
    state5_Qp5 = state5_Qp05-Ssi12
    state6_Qp6 = state6_Qp06-Ssi12
    state7_Qp7 = state7_Qp07-Ssi12
    state8_Qp8 = state8_Qp08-Ssi12
    state9_Qp9 = state9_Qp09-Ssi12
    state10_Qp10 = state10_Qp010-Ssi12
    state11_Qp11 = state11_Qp011-Ssi12
    state12_Qp12 = state12_Qp012-Ssi12
    list_Qp = [state1_Qp1, state2_Qp2, state3_Qp3, state4_Qp4, state5_Qp5, state6_Qp6, state7_Qp7, state8_Qp8,state9_Qp9, state10_Qp10, state11_Qp11, state12_Qp12]


    state13_FC = np.var(list_Qp)

    write_csv2(state1_Qp1, state2_Qp2, state3_Qp3, state4_Qp4, state5_Qp5, state6_Qp6, state7_Qp7, state8_Qp8,state9_Qp9, state10_Qp10, state11_Qp11, state12_Qp12, state13_FC)  # write_aa.csv
    nStep = nStep + 1
    bb = nStep - 1
    write_csv1(bb)  # write_bb.csv
    print(nStep - 1)
    while True:
        try:
            dd = read__csv1()[0][0]    # read_dd.csv
        except:
            continue

        if dd == nStep-1 :
            break
        else:
            continue



while 1 == 1:
    if __name__ == "__main__":
        main()





