
import os


class VideoDataManager:

    def __init__(self, file_path):

        self.file_path = file_path
        self.file_name = os.path.basename(file_path)
        self.file = open(self.file_path, 'r')
        self.data = {} 
        
    def __del__(self):
        
        self.file.close()
    
    def get_video_name(self):
        
        first_line = self.file.readline()
        return first_line.split(',')[0].split('_')
    
    def get_frame_number(self, line):
        
        return line.split(',')[0].split('_')[-1]
    
    def get_float_values(self):
        
        for line in file:
            line = line.strip()
            if line:
                
                parts = line.split(',')
                if len(parts) < 2:
                    print(f"خطا: خط {line_num} فرمت صحیحی ندارد")
                    continue
                
                # استخراج نام فریم (مثل: vid1202_frame_18)
                frame_name = parts[0].strip()
                
                # استخراج نام ویدیو و شماره فریم
                # فرمت: vid[شماره]_frame_[شماره]
                try:
                    video_name = frame_name.split('_frame_')[0]  # مثلاً: vid1202
                    frame_number = int(frame_name.split('_frame_')[1])  # مثلاً: 18
                except (IndexError, ValueError):
                    print(f"خطا: نام فریم '{frame_name}' فرمت صحیحی ندارد")
                    continue
                
                # تبدیل مقادیر به float
                try:
                    features = [float(val.strip()) for val in parts[1:]]
                except ValueError as e:
                    print(f"خطا در تبدیل مقادیر خط {line_num}: {e}")
                    continue
                
                # ذخیره اطلاعات
                self.data.append({
                    'video_name': video_name,
                    'frame_number': frame_number,
                    'frame_name': frame_name,
                    'features': features
                })
        
        return self.data
    
    def get_all_video_names(self):

        video_name = self.get_video_name()
        return [video_name] if video_name else []
    
    def get_all_frame_numbers(self):

        if not self.data:
            self.get_float_values()
        
        return [item['frame_number'] for item in self.data]
    
