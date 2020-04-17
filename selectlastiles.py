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
        print(str(seg) + ' file: '+str(file_number)+' of '+str(number_files))
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

    snip_files(adress, file_path,output_directory, maps[i], marge)

print("--- RUNTIME --- %s seconds ---" % (time.time()-start_time))