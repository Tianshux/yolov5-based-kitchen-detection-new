# import os
# import shutil
# # 图片文件夹和标注文件夹路径
# image_folder = r'C:\Users\A\Desktop\phone_data\JPEGImages'
# annotation_folder = r'C:\Users\A\Desktop\3000-4500'
# destination_folder = r'C:\Users\A\Desktop\new_phone\JPEGImages'
# # 获取图片文件夹和标注文件夹中的所有文件名
# image_files = os.listdir(image_folder)
# annotation_files = os.listdir(annotation_folder)
#
# # 确保文件名列表按照字母顺序排列，以便一一对应
# image_files.sort()
# annotation_files.sort()
# # 检查图片文件和标注文件的数量是否一致
# for xml_name in annotation_files:
#     for i in range(len(image_files)):
#         if(xml_name[0:4] == image_files[i][0:4]):
#             old_image_path = os.path.join(image_folder, image_files[i])
#             new_image_path = os.path.join(image_folder, f'{i+514}.jpg')
#             old_annotation_path = os.path.join(annotation_folder, xml_name)
#             new_annotation_path = os.path.join(annotation_folder, f'{i+514}.txt')
#             # 重命名图片文件
#             os.rename(old_image_path, new_image_path)
#             # 重命名标注文件
#             os.rename(old_annotation_path, new_annotation_path)
#             shutil.move(old_image_path, destination_folder)
#             print(f"重命名：{old_image_path} -> {new_image_path}")
#             print(f"重命名：{old_annotation_path} -> {new_annotation_path}")

# for i in range(len(annotation_files)):
#     old_annotation_path = os.path.join(annotation_folder, annotation_files[i])
#     new_annotation_path = os.path.join(annotation_folder, f'{i + 514}.xml')
#     os.rename(old_annotation_path, new_annotation_path)
#     print(f"重命名：{old_annotation_path} -> {new_annotation_path}")

import os

# 文件夹路径
folder_path = r'C:\Users\A\Desktop\厨师服\Annotations'

# 遍历文件夹中的所有文件
for file_name in os.listdir(folder_path):
    # 构建原始文件路径
    old_file_path = os.path.join(folder_path, file_name)

    # 判断是否是文件
    if os.path.isfile(old_file_path):
        # 获取文件名和文件扩展名
        base_name, extension = os.path.splitext(file_name)

        # 构建新的文件名，在原始文件名后加上下划线
        new_file_name = f"{base_name}_"

        # 构建新的文件路径
        new_file_path = os.path.join(folder_path, new_file_name + extension)

        # 重命名文件
        os.rename(old_file_path, new_file_path)

        print(f"文件重命名：{old_file_path} -> {new_file_path}")
