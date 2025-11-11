import sys

ModulePath = '/home/ahmad/Documents/phd-courses/phd-courses/deep-network/HomeWorks/1/vids_feats'

if ModulePath not in sys.path:
    sys.path.append(ModulePath)

import glob
from module import DataManagement

files_directory_path = "./vids_feats"
path_list = glob.glob(files_directory_path + "/*.txt")

"""Finally, We want to have a list containing 4 dictionaries 
for 4 videos that contain the information of video frames."""

final_frame_list = list()

for file_path in path_list:
    
    dic1 = {}
    file = open(file_path,'r')
    print("********************************************\n")
    print("file_name")
    Manager = DataManagement(file_path)
    video_name_flag=1
    
    for line in file:
        
        if video_name_flag == 1:
            video_name = Manager.get_video_file_name(line)
            video_name_flag = 0
            print("this is video name: %s \n" % video_name)
        frame_num = Manager.get_frame_number(line)
        print('frame number : %s \n' % frame_num)
        AnnotationList = Manager.make_float_frame_data(line)
        dic1[frame_num] = AnnotationList.copy()
    final_frame_list.append(dic1)

print('List of Dictionaries:\n',final_frame_list)