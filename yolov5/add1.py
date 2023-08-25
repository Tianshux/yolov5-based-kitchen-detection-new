import os

accept_list = ['8', '9']
def add_prefix_to_lines(folder_path, prefix):
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith('.txt') and file_name.startswith("mouse"):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r') as file:
                lines = file.readlines()
            with open(file_path, 'w') as file:
                for i, line in enumerate(lines):
                    if line.startswith("10"):
                        print(line)
                        file.write(str(int(line[:2]) - 6) + line[2:])

if __name__ == "__main__":
    folder_path = r"C:\Users\A\Desktop\pure_chef_mouse\Annotations"  # 替换为你的文件夹路径
    prefix = "1"
    add_prefix_to_lines(folder_path, prefix)
