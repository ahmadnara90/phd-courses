
import os
import glob
from video_data_manager import VideoDataManager


def main():

    directory_path = "../vids_feats"
    

    file_list = glob.glob(os.path.join(directory_path, "*.txt"))
    
    for idx, file_path in enumerate(file_list, 1):
        print(idx,file_path)
    
    

    video_managers = []
    
    for file_path in file_list:
        print("=" * 70)
        print("is processing: ",file_path)
        print("=" * 70)
        

        manager = VideoDataManager(file_path)

        data = manager.get_float_values()
        
        if data:

            first_frame = data[0]
            print(f"\n files information:")
            print(f"   - video name: {manager.get_video_name()}")
            print(f"   - number of all frames: {len(data)}")
            print(f"   - frames limitation: {min(manager.get_all_frame_numbers())} - {max(manager.get_all_frame_numbers())}")
            print(f"   - number of features in each frame: {len(first_frame['features'])}")
            
            print(f"\n🎬 نمونه‌ای از فریم اول:")
            print(f"   - نام فریم: {first_frame['frame_name']}")
            print(f"   - شماره فریم: {first_frame['frame_number']}")
            print(f"   - 5 ویژگی اول: {first_frame['features'][:5]}")
            
            print(f"\n📋 چند فریم نمونه:")
            for i, frame in enumerate(data[:3], 1):  # نمایش 3 فریم اول
                print(f"   {i}. فریم {frame['frame_number']}: {frame['frame_name']}")
        

        print()
        print(manager)
        

        video_managers.append(manager)
        print()
    

    print("=" * 70)
    print("📈 خلاصه کلی:")
    print("=" * 70)
    print(f"تعداد کل فایل‌های پردازش شده: {len(video_managers)}")
    
    total_frames = sum(len(manager.data) for manager in video_managers)
    print(f"تعداد کل فریم‌ها در همه فایل‌ها: {total_frames}")

    print(f"تعداد ویدیوها: {len(video_managers)}")
    print(f"نام‌های ویدیو: {', '.join([m.get_video_name() for m in video_managers])}")
    
    print("\n📁 جزئیات هر فایل:")
    for manager in video_managers:
        num_frames, num_features = manager.get_data_shape()
        video_name = manager.get_video_name()
        print(f"  - {manager.file_name} ({video_name}): {num_frames} فریم × {num_features} ویژگی")
    
    print("\n" + "=" * 70)
    print("✅ پردازش با موفقیت انجام شد!")
    print("=" * 70)


if __name__ == "__main__":
    main()
