import os

def remove_first_one(delete_file, delete_src):
    with open(delete_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if(len(line) < 3):
                continue
            for file_name in os.listdir(delete_src):
                if(file_name.startswith(line[:-1])):
                    os.remove(delete_src + "\\" + file_name)
                    print(f'delete {delete_src +file_name} successfully')

    # for file_name in os.listdir(folder_path):
    #     if file_name.lower().endswith('.txt') and file_name.lower().startswith('mouse'):
    #         file_path = os.path.join(folder_path, file_name)
    #         with open(file_path, 'r') as file:
    #             lines = file.readlines()
    #         with open(file_path, 'w') as file:
    #             for i, line in enumerate(lines):
    #                 if i == 0:
    #                     # 删除第一行的第一个字符"1"
    #                     line = line[1:]
    #                 file.write(line)

if __name__ == "__main__":
    delete_file = r"C:\Users\A\Desktop\new_all\delete_txt.txt"  # 替换为你的文件夹路径
    delete_src = r"C:\Users\A\Desktop\new_all\Annotations"
    remove_first_one(delete_file, delete_src)

