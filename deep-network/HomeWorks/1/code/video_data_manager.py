
import os


class VideoDataManager:
    def __init__(self, file_path):

        self.file_path = file_path
        self.file_name = os.path.basename(file_path)

        #self._file = open(self.file_path, 'r')
        self._parse_file()
    
    def _parse_file(self):
        
        self._frame_data = {}
        self._video_name = None

        with open(self.file_path,'r') as file:
            for idx,line in enumerate(file):
                
                parts = line.split(',')
                frame_info = parts[0]

                #find name of video
                if idx ==0:
                    self._video_name = frame_info.split('_')[0]
                
                #frame number
                frame_number = frame_info.split('_')[-1]

                #features
                features = [float(j) for j in line.split(',')[1:]]

                self._frame_data[frame_number] = features

    
    
    def get_video_name(self):

        return self._video_name

    def get_float_frames_dict(self):
       
        return self._frame_data.copy()
    
    def get_all_frame_numbers(self):

        return list(self._frame_data.keys())
    