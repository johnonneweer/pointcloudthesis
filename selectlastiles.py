import sys, os, subprocess
import pandas as pd
import time

#timeit
start_time = time.time()

#function
def snip_files(adress, file_path, output_directory, seg, marge):
    file_number = 0
    for index, row in adress.iterrows():
        file_number += 1
        # print(str(seg) + ' file: '+str(file_number)+' of '+str(number_files))
        # print(row[0], row[1], row[2])
        min_x = row[1] - marge
        max_x = row[1] + marge
        min_y = row[2] - marge
        max_y = row[2] + marge

        output_name = int(row[0])
        command = []
        #file path is the path to the file
        command.append("las2txt")
        command.append("-i")
        command.append(file_path)

        command.append("-parse xyzRGBcia")

        #choose seperator
        command.append("-sep comma")
        #Rd values
        command.append("-inside")
        command.append(min_x)
        command.append(min_y)
        command.append(max_x)
        command.append(max_y)

        #output_directory is de output directory
        command.append("-odir")
        command.append(output_directory)

        #output_name moet de naam van de postcode huisnum combinatie
        command.append("-o")
        command.append(output_name)
        
        command_line = " ".join([str(elem) for elem in command])
        os.system(command_line)

def snip_tiles(adress, file_path, output_directory, seg, marge):
    file_number = 0
    for p in range(len(adress[2])):
        file_number += 1
        print(str(seg) + ' file: '+str(file_number)+' of '+str(len(adress[2])))
        # print(row[0], row[1], row[2])
        min_x = adress[2][p][0]
        max_x = adress[2][p][0] + marge
        min_y = adress[2][p][1]
        max_y = adress[2][p][1] + marge

        output_name = str(adress[0]) +'_'+ str(adress[2][p][0]) +'_'+ str(adress[2][p][1])
        print(output_name)
        command = []
        #file path is the path to the file
        command.append("las2txt")
        command.append("-i")
        command.append(file_path)

        command.append("-parse xyzRGBcia")

        #choose seperator
        command.append("-sep comma")
        #Rd values
        command.append("-inside")
        command.append(min_x)
        command.append(min_y)
        command.append(max_x)
        command.append(max_y)

        #output_directory is de output directory
        command.append("-odir")
        command.append(output_directory)

        #output_name moet de naam van de postcode huisnum combinatie
        command.append("-o")
        command.append(output_name)
        
        command_line = " ".join([str(elem) for elem in command])
        os.system(command_line)

import numpy as np

area_length = 1000
tile_size = 25

x = int(area_length / tile_size)
t = np.zeros([2,x,x])

for j in range(x):
    for i in range(x):
        t[0][i][j] = i
        t[1][j][i] = i
tiles = np.stack((t[0],t[1]), axis=2).reshape(1,x*x,2).astype(int)[0]*tile_size

files = ['S_25GN1.LAZ', 'S_25GZ2.LAZ', 'S_31HZ2.LAZ', 'S_26CN2.LAZ', 'S_20AN1.LAZ']
areas = ['ams','azo', 'utr', 'alm', 'hoo']
coordinates = [[120900, 484590],[127400,478300],[137150,454750],[145040,486000],[140250, 522500]]

tiles_2 = np.zeros((len(coordinates),tiles.shape[0],tiles.shape[1])).astype(int)

for i in range(len(coordinates)):
    tiles_2[i] = coordinates[i] + tiles

total = list(zip(areas,files,tiles_2)) #0 area 1 file 2 coordinate array
print(len(total[0][2]))

total = [total[3]]
print(total[0])
# sys.exit()
# snip_files(adress, file_path, output_directory, seg, marge)

os.chdir(r"C:\Users\Sustainables\Documents\Thesis\LAStools\bin")

for i in range(len(total)):
    file_path = r"C:\Users\Sustainables\Documents\Thesis"
    file_path = file_path + '\\' + total[i][1]
    output_directory = r"C:\Users\Sustainables\Documents\Thesis\Data\AHN3"
    output_directory = output_directory + '\\' + total[i][0]
    # output_directory.mkdir(exist_ok=True)
    snip_tiles(total[i], file_path, output_directory, total[i][0], tile_size)

sys.exit()

#Program Settings

os.chdir(r"C:\Users\Sustainables\Documents\Thesis\LAStools\bin")
number_files = 10
marge = 30

maps = ['\pijp', '\gein', '\hoogkarspel', r"\utrecht"]
files = ['\S_25GN1.LAZ', '\S_25GZ2.LAZ', '\S_20AN1.LAZ', '\S_31HZ2.LAZ']
lists = ['\list_pijp1073', '\list_gein1106', '\list_hoogkarspel', '\list_utrecht3582']

maps = [r'\test']
files = ['\S_26AZ2.LAZ']
lists = ['\list_almere']




for i in range(len(maps)):
    file_path = r"C:\Users\Sustainables\Documents\Thesis"
    file_path = file_path + files[i]
    output_directory = r"C:\Users\Sustainables\Documents\Thesis\Data\AHN3"
    output_directory = output_directory + maps[i]
    list_path = r"C:\Users\Sustainables\Documents\Thesis\Data\AHN3"
    list_path = list_path + lists[i] + '.txt'
    
    adress = pd.read_csv(list_path, sep='\\t', engine='python')
    adress = adress.drop_duplicates(subset=['BuildingId'], keep = 'first', inplace=False)
    adress = adress.sample(n = number_files)
    adress = adress.iloc[:,[0, 1, 2]]
    snip_files(adress, file_path, output_directory, seg, marge)
    snip_files(adress, file_path,output_directory, maps[i], marge)

print("--- RUNTIME --- %s seconds ---" % (time.time()-start_time))