class DataManagement():
    def __init__(self, FilePath):
        self.filepath = FilePath
        
    def get_video_file_name(self,line):
        name = line.split(',')[0].split('_')[0]
        return name
    def get_frame_number(self,line):
        frame_num = line.split(',')[0].split('_')[-1]
        return frame_num
    def make_float_frame_data(self,line):
        return [float(j) for j in line.split(',')[1:]] 

        