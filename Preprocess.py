# coding: utf-8
import os
import argparse
import cv2
import subprocess
import multiprocessing


def extract_videos(data_dir, vid_name, root_audio, root_frame, fps, audio_rate):
    video_file_path = os.path.join(data_dir, vid_name)
    vid_id = vid_name.split('.')[0]
    # extract audio
    audio_file_path = os.path.join(root_audio, vid_id + ".wav")
    if not os.path.exists(os.path.dirname(audio_file_path)):
        os.makedirs(os.path.dirname(audio_file_path)) 
    aud_command = ["ffmpeg", "-loglevel", "warning",  "-i", video_file_path, "-ab", "160k", "-ac", "1", "-ar", "11025", "-vn",audio_file_path]
    #command is: ffmpeg -i yy2vL2RUiPI.mp4 -ab 160k -ar 11025 -vn test.wav
    subprocess.call(aud_command)
    print("audio processing is over, next is video processing part")
    # extract video
    video_file_path_fps = os.path.join(data_dir, vid_id + '_fps.mp4')
    vid_command = ["ffmpeg", "-loglevel", "warning", "-i", video_file_path, "-r", "8", video_file_path_fps]
    subprocess.call(vid_command)
    count = 1
    cap = cv2.VideoCapture(video_file_path_fps)
    #cap_fps = cap.get(cv2.CAP_PROP_FPS)
    #assert cap_fps == fps
    print("creating frames.....\n\n")
    while 1:
        # get a frame
        frame_file_path = os.path.join(root_frame, vid_name, '{:06d}.jpg'.format(count))
        if not os.path.exists(os.path.dirname(frame_file_path)):
            os.makedirs(os.path.dirname(frame_file_path))
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        cv2.imwrite(frame_file_path, frame)
    cap.release()
    os.remove(video_file_path_fps)
    print('I am done!')
   

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./data_all',
                        help="data dir which you want to save")
    parser.add_argument('--root_audio', default='./data/audio',
                        help="root for extracted audio files")
    parser.add_argument('--root_frame', default='./data/frames',
                        help="root for extracted video frames")
    parser.add_argument('--fps', default=8, type=int,
                        help="fps of video frames")
    parser.add_argument('--audio_rate', default=11.025, type=float,
                        help="rate of audio")
    args = parser.parse_args()
    if not os.path.exists(args.root_audio):
        os.makedirs(args.root_audio)
    if not os.path.exists(args.root_frame):
        os.makedirs(args.root_frame)

    # use multiprocessing pool
    pool = multiprocessing.Pool(2)
    for vid_name in os.listdir(args.data_dir):
        pool.apply_async(extract_videos, args=(
            args.data_dir, vid_name, args.root_audio, args.root_frame, args.fps, args.audio_rate))
    pool.close()
    pool.join()
