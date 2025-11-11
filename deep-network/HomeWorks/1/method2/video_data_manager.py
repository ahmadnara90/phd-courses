
import os


class VideoDataManager:

    def __init__(self, file_path):

        self.file_path = file_path
        self.file_name = os.path.basename(file_path)
        self.data = {} 
        
    def get_video_name(self):

        if not self.data:
            self.get_float_values()
        
        if self.data:
            return self.data[0]['video_name']
        return None
    
    def get_frame_number(self, line):
        
        return line.split(',')[0].split('_')[-1]
    
    def get_float_values(self):

        self.data = {}
        
        try:
            with open(self.file_path, 'r') as file:
                for line_num, line in enumerate(file, 1):
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
                        
        except FileNotFoundError:
            print(f"خطا: فایل {self.file_path} پیدا نشد!")
        except Exception as e:
            print(f"خطای غیرمنتظره: {e}")
        
        return self.data
    
    def get_all_video_names(self):
        """
        دریافت نام ویدیو (در هر فایل فقط یک ویدیو وجود دارد)
        
        Returns:
            list: لیست شامل نام ویدیو
        """
        video_name = self.get_video_name()
        return [video_name] if video_name else []
    
    def get_all_frame_numbers(self):
        """
        دریافت لیست تمام شماره‌های فریم موجود در فایل
        
        Returns:
            list: لیست شماره‌های فریم
        """
        if not self.data:
            self.get_float_values()
        
        return [item['frame_number'] for item in self.data]
    
    def get_data_shape(self):
        """
        بدست آوردن ابعاد داده‌ها
        
        Returns:
            tuple: (تعداد فریم‌ها، تعداد ویژگی‌ها)
        """
        if not self.data:
            self.get_float_values()
        
        if self.data:
            return (len(self.data), len(self.data[0]['features']) if self.data else 0)
        return (0, 0)
    
    def get_frame_by_number(self, frame_number):
        """
        دریافت اطلاعات یک فریم خاص بر اساس شماره فریم
        
        Args:
            frame_number (int): شماره فریم مورد نظر
            
        Returns:
            dict: اطلاعات فریم یا None
        """
        if not self.data:
            self.get_float_values()
        
        for item in self.data:
            if item['frame_number'] == frame_number:
                return item
        return None
    
    def __str__(self):
        """
        نمایش اطلاعات شی به صورت رشته
        """
        if not self.data:
            self.get_float_values()
        
        video_name = self.get_video_name()
        num_frames, num_features = self.get_data_shape()
        frame_numbers = self.get_all_frame_numbers()
        
        return (f"VideoDataManager(\n"
                f"  فایل: {self.file_name}\n"
                f"  نام ویدیو: {video_name}\n"
                f"  تعداد فریم‌ها: {num_frames}\n"
                f"  تعداد ویژگی‌ها در هر فریم: {num_features}\n"
                f"  محدوده شماره فریم‌ها: {min(frame_numbers) if frame_numbers else 0} - "
                f"{max(frame_numbers) if frame_numbers else 0}\n"
                f")")
