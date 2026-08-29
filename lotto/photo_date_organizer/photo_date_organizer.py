from datetime import datetime
import os
import shutil

source_folder = 'Aqua_Planet_Jeonggeun'  # 원본 사진 폴더 경로
destination_folder = 'photos_by_date'  # 정리된 폴더가 위치할 경로


def organize_photos_by_date(source_folder, destination_folder):
    # Source folder에 있는 모든 파일을 가져옴
    for filename in os.listdir(source_folder):
        # 파일의 전체 경로
        file_path = os.path.join(source_folder, filename)

        # 파일인지 확인 (폴더 제외)
        if os.path.isfile(file_path):
            # 파일의 생성 날짜 가져오기 (변경하려면 os.path.getmtime 등으로 수정 가능)
            creation_time = os.path.getctime(file_path)
            date_folder_name = datetime.fromtimestamp(creation_time).strftime('%Y-%m-%d')

            # 해당 날짜에 맞는 폴더 경로 생성
            date_folder_path = os.path.join(destination_folder, date_folder_name)
            os.makedirs(date_folder_path, exist_ok=True)

            # 파일을 날짜 폴더로 이동
            shutil.move(file_path, os.path.join(date_folder_path, filename))
            print(f'{filename} -> {date_folder_path}')


def main():
    print('organize_photos_by_date Start!!')

    organize_photos_by_date(source_folder, destination_folder)


if __name__ == '__main__':
    print('__file__'+'start!!')
    main()
    print('__file__'+'Done!!')
