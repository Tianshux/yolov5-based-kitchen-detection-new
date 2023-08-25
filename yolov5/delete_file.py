import os
import shutil
# 图片文件夹和标注文件夹路径
image_folder = r'C:\Users\A\Desktop\phone_data\JPEGImages'
annotation_folder = r'C:\Users\A\Desktop\3000-4500'
destination_folder = r'C:\Users\A\Desktop\new_phone\JPEGImages'
# 获取图片文件夹和标注文件夹中的所有文件名
image_files = os.listdir(image_folder)
annotation_files = os.listdir(annotation_folder)

# 确保文件名列表按照字母顺序排列，以便一一对应
image_files.sort()
annotation_files.sort()
# 检查图片文件和标注文件的数量是否一致
for xml_name in annotation_files:
    for i in range(len(image_files)):
        if(xml_name[0:4] == image_files[i][0:4]):
            old_image_path = os.path.join(image_folder, image_files[i])
            new_image_path = os.path.join(image_folder, f'{i+514}.jpg')
            old_annotation_path = os.path.join(annotation_folder, xml_name)
            new_annotation_path = os.path.join(annotation_folder, f'{i+514}.txt')
            # 重命名图片文件
            os.rename(old_image_path, new_image_path)
            # 重命名标注文件
            os.rename(old_annotation_path, new_annotation_path)
            shutil.move(old_image_path, destination_folder)
            print(f"重命名：{old_image_path} -> {new_image_path}")
            print(f"重命名：{old_annotation_path} -> {new_annotation_path}")

for i in range(len(annotation_files)):
    old_annotation_path = os.path.join(annotation_folder, annotation_files[i])
    new_annotation_path = os.path.join(annotation_folder, f'{i + 514}.xml')
    os.rename(old_annotation_path, new_annotation_path)
    print(f"重命名：{old_annotation_path} -> {new_annotation_path}")