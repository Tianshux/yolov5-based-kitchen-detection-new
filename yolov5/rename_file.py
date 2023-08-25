import os


def rename_images(folder_path):
    counter = 0

    for filename in os.listdir(folder_path):
        if is_image_file(filename):
            extension = get_file_extension(filename)
            new_filename = f"{counter:04d}.{extension}"
            new_filepath = os.path.join(folder_path, new_filename)
            old_filepath = os.path.join(folder_path, filename)

            try:
                os.rename(old_filepath, new_filepath)
                print(f"Renamed {filename} to {new_filename}")
            except Exception as e:
                print(f"Failed to rename {filename}: {str(e)}")

            counter += 1


def is_image_file(filename):
    extension = get_file_extension(filename)
    return extension.lower() in ["jpg", "jpeg", "png", "gif", "webp"]


def get_file_extension(filename):
    return os.path.splitext(filename)[1][1:].lower()


folder_path = r"C:\Users\A\Desktop\厨师服\厨师服_-_Bing_images"  # 文件夹路径
rename_images(folder_path)
