import sys
import os

ModulePath = os.path.abspath(os.getcwd())

if ModulePath not in sys.path:
    sys.path.append(ModulePath)



import glob
from video_data_manager import VideoDataManager


def main():

    directory_path = "../vids_feats"
    

    file_list = glob.glob(os.path.join(directory_path, "*.txt"))
    


    for idx, file_path in enumerate(file_list, 1):
        print(idx,file_path)
    
    """
    A list of videos dictionary after make float feature numbers.
    We save each video's frames features in a dictionary where
    the keys are the frame numbers and the values are the list of float features.
    Each dictionary is appended to the videos_after_modify list.

    """

    videos_after_modify = []
    
    for file_path in file_list:

        manager = VideoDataManager(file_path)

        print("\n"+"=" * 70)
        print("is processing: ",manager.file_name)
        print("=" * 70)
        
        print("video name: ", manager.get_video_name())
        frames_features = manager.get_float_frames_dict()
        
        print("frame_numbers :", manager.get_all_frame_numbers())

        videos_after_modify.append(frames_features)
    

    print("\n"+"=" * 70)
    print("Videos processed(dictionary saved number): ", len(videos_after_modify))
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
