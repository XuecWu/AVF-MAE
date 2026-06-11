import os
import glob
import time
import shutil
import subprocess
import concurrent.futures
from joblib import Parallel, delayed
from tqdm import tqdm
OPENFACE_EXE = './preprocess/face_and_audio/OpenFace_2.2.0_win_x64/FeatureExtraction.exe'


def process_one_video(video_file, in_dir, out_dir, openface_exe=OPENFACE_EXE, img_size=112):

    file_name = os.path.splitext(video_file.replace(in_dir + '\\', ''))[0]
    print(file_name)
    out_dir   = os.path.join(out_dir, file_name)

    if os.path.exists(out_dir):
        print(f'Note: "{out_dir}" already exist!')
        return video_file
    else:
        os.makedirs(out_dir)

    cmd = f'"{openface_exe}" -f "{video_file}" -out_dir "{out_dir}" -simalign -simsize {img_size} -format_aligned jpg -nomask'

    subprocess.call(cmd, shell=False)

    return video_file


def main(video_dir, out_dir, openface_exe=OPENFACE_EXE, multi_process=True, video_template_path='*.mp4', img_size=112):


    video_files = glob.glob(os.path.join(video_dir, video_template_path))

    n_files     = len(video_files)
    print(f'Total videos: {n_files}.')

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    start_time = time.time()


    if multi_process:
        Parallel(n_jobs=8)(delayed(process_one_video)(video_file, video_dir, out_dir, openface_exe, img_size) for video_file in tqdm(video_files))
    else:
        for i, video_file in enumerate(video_files, 1):
            print(f'Processing "{os.path.basename(video_file)}"...')
            process_one_video(video_file, video_dir, out_dir, openface_exe, img_size)
            print(f'"{os.path.basename(video_file)}" done, rate of progress: {100.0 * i / n_files:3.0f}% ({i}/{n_files})')

    end_time = time.time()
    print('Time used for video face extraction: {:.1f} s'.format(end_time - start_time))


def copy_one_video(src_dir, tgt_dir):
    shutil.copytree(src_dir, tgt_dir)
    print(f'Copy "{src_dir}" to "{tgt_dir}"')


if __name__ == '__main__':


    dataset_root = './preprocess/face_and_audio'
    video_dir    = os.path.join(dataset_root, 'Face_test')
    img_size     = 256


    file_ext            = 'mp4'
    video_template_path = f'*.{file_ext}'
    out_dir             = os.path.join(video_dir, '../openface')


    main(video_dir, out_dir, video_template_path=video_template_path, multi_process=False, img_size=img_size)


    src_root = out_dir
    tgt_root = out_dir.replace('openface', 'face_aligned')
    count    = 0
    src_dirs, tgt_dirs = [], []

    for sample_dir in os.scandir(src_root):
            sample_name = sample_dir.name

            tgt_dir     = os.path.join(tgt_root, sample_name)
            src_dir     = os.path.join(sample_dir, f'{sample_name}_aligned')

            src_dirs.append(src_dir)
            tgt_dirs.append(tgt_dir)
            count += 1

    print(f'Total videos: {count}.')
    Parallel(n_jobs=16)(delayed(copy_one_video)(src_dir, tgt_dir) for src_dir, tgt_dir in tqdm(zip(src_dirs, tgt_dirs)))
