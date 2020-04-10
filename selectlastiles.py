import sys, os, subprocess
import pandas as pd

# this defines the LAStoos\bin\las2txt.exe path and writes it to an array
lastools_path = r"C:\Users\Sustainables\Documents\Thesis\LAStools\bin"

las2txt_path = lastools_path+"\\las2txt.exe"

#START
command = ["las2txt"]

# add input LiDAR
command.append("-i")
command.append(' '+'inputtest')

adress = pd.read_csv("/Users/john/Downloads/files_almere.csv")
#adress = adress.drop_duplicates(subset='BuildingId', keep = 'first', inplace=True)

adress = adress.sample(n = 5)
adress = adress.iloc[:,[2, 4, 5]]



# add parameters

print("HET BEGIN")

file_path = r"C:\Users\Sustainables\Documents\Thesis\S_26AZ2.LAZ"
output_directory = r"C:\Users\Sustainables\Documents\Thesis\Data\AHN3\almere"

for index, row in adress.iterrows():
    print("next file")
    print(index)
    # print(row[0], row[1], row[2])
    min_x = row[1] - 10
    max_x = row[1] + 10
    min_y = row[2] - 10
    max_y = row[2] + 10

    output_name = int(row[0])
    command = []
    #file path is the path to the file
    command.append("lasview")
    command.append("-i")
    command.append(file_path)

    command.append("-parse xyzRGBcia")

    #choose seperator
    command.append("-sep comma")
    #Rd values
    command.append("-keep_xy")
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

    # print(command)
    
    command_line = " ".join([str(elem) for elem in command])
    print(command_line)
    os.system(command_line)


# sys.exit()

#finally run las2txt
#process = subprocess.Popen(command)

# command_length = len(command)
# command_string = str(command[0])
# command[0] = command[0].strip('"')
# for i in range(1, command_length):
#     command_string = command_string + " " + str(command[i])
#     command[i] = command[i].strip('"')


#RdX RdY PostCode
# df_test = 

# go to C:\Users\Sustainables\Documents\Thesis\LAStools\bin

#os.system("cd C:\Users\Sustainables\Documents\Thesis\LAStools\bin")


# min_x = 147014
# max_x = 147034
# min_y = 490087
# max_y = 490107


# file_path = "file_path"
# output_directory = "output_dir"
# output_name = "output_name"

# #file path is the path to the file
# command.append("lasview")
# command.append("-i")
# command.append(file_path)

# command.append("-parse xyzRGBcia")

# #choose seperator
# command.append("-sep comma")

# #Rd values
# command.append("-keep_xy")
# command.append(min_x)
# command.append(min_y)
# command.append(max_x)
# command.append(max_y)

# #output_directory is de output directory
# command.append("-odir")
# command.append(output_directory)

# #output_name moet de naam van de postcode huisnum combinatie
# command.append("-o")
# command.append(output_name)

# print(command)

# command_line = " ".join([str(elem) for elem in command])
# print(command_line)

# os.system("cd ~/AI")

#os.system("pip install")

