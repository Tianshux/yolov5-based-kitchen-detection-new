import os

def find_line_starting_with_11(folder_path):
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith('.txt') and file_name.startswith("mouse"):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r') as file:
                for line_number, line in enumerate(file, start=1):
                    if line.strip().startswith("7"):
                        print(f'文件：{file_name}，第{line_number}行：{line.strip()}')

if __name__ == "__main__":
    folder_path = r"C:\Users\A\Desktop\new_all\Annotations"  # 替换为你的文件夹路径
    find_line_starting_with_11(folder_path)