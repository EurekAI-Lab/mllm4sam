import os
import pyperclip

class DirectoryCodeExporter:
    def __init__(self, paths, output_txt, ignore_dirs=None, empty_file=True, copy_to_clipboard=True):
        """
        Initialize the exporter with:
        paths: A list of directory or file paths to search for .py and .yaml files.
        output_txt: The output file path to store the concatenated code.
        ignore_dirs: A list of directory paths to ignore during the search.
        empty_file: Boolean, if True, overwrite the file; if False, append to it.
        copy_to_clipboard: Boolean, if True, copy the result to clipboard.
        """
        self.paths = paths if isinstance(paths, list) else [paths]
        self.output_txt = output_txt
        self.empty_file = empty_file
        self.ignore_dirs = set(os.path.abspath(dir_path) for dir_path in ignore_dirs) if ignore_dirs else set()
        self.copy_to_clipboard = copy_to_clipboard

    def export_to_txt(self):
        file_paths = []

        # 遍历所有路径
        for path in self.paths:
            abs_path = os.path.abspath(path)
            if os.path.isdir(abs_path):
                for root, dirs, files in os.walk(abs_path):
                    abs_root = os.path.abspath(root)
                    dirs_to_remove = []
                    for dir_name in dirs:
                        dir_path = os.path.join(abs_root, dir_name)
                        abs_dir_path = os.path.abspath(dir_path)
                        if any(abs_dir_path == ignore_dir or abs_dir_path.startswith(ignore_dir + os.sep) for ignore_dir in self.ignore_dirs):
                            print(f"Ignoring directory: {abs_dir_path}")
                            dirs_to_remove.append(dir_name)
                    for dir_name in dirs_to_remove:
                        dirs.remove(dir_name)
                    for file in files:
                        if file.endswith('.py') or file.endswith('.yaml'):
                            file_paths.append(os.path.join(abs_root, file))
            elif os.path.isfile(abs_path):
                if abs_path.endswith('.py') or abs_path.endswith('.yaml') or abs_path.endswith('.h') or abs_path.endswith('.cpp') or abs_path.endswith('.cu'):
                    file_dir = os.path.dirname(abs_path)
                    if not any(os.path.abspath(file_dir).startswith(ignore_dir + os.sep) for ignore_dir in self.ignore_dirs):
                        file_paths.append(abs_path)
                    else:
                        print(f"Skipping file {abs_path} as it is inside an ignored directory.")
                else:
                    print(f"Skipping file {abs_path}, not a .py or .yaml file.")
            else:
                print(f"Path {abs_path} does not exist.")

        print(f"Found {len(file_paths)} files to process.")

        mode = 'w' if self.empty_file else 'a'
        all_code = ""

        with open(self.output_txt, mode, encoding='utf-8') as outfile:
            if not self.empty_file:
                outfile.write('\n')
            for file_path in file_paths:
                if os.path.exists(file_path) and (file_path.endswith('.py') or file_path.endswith('.yaml')):
                    print(f"Processing file: {file_path}")
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        code = infile.read()
                        outfile.write(f"# {file_path}\n")
                        outfile.write(code)
                        outfile.write('\n\n')
                        outfile.write('###################################################################\n\n')
                        all_code += f"# {file_path}\n{code}\n\n" \
                                    '#################################################################\n\n'
                else:
                    print(f"File {file_path} does not exist or is not a Python/YAML file.")

            # 添加代码建议提示
            code_suggestion = ('<Code requirement>\n'
                               'Help me with this problem. If you change my code, you should give me full class of that part,\n'
                               'You should never ignore, simplify, or skip code, NEVER ignore my demand.\n'
                               'Your code should be always complete and runnable. Eligible for top conference, such as CVPR, AAAI, etc. Highly maintanable and innovative.\n'
                               'For hard shape mismatching problem, you can always print out for help you debug.\n'
                               'Never return me a snippet of code.'
                               )
            outfile.write(code_suggestion)
            all_code += code_suggestion

        if self.copy_to_clipboard:
            try:
                pyperclip.copy(all_code)
                print(f"All files have been written to {self.output_txt} and copied to clipboard.")
            except pyperclip.PyperclipException as e:
                print("警告：无法将内容复制到剪贴板。")
                print(e)
                print(f"所有文件已写入 {self.output_txt}。")
        else:
            print(f"所有文件已写入 {self.output_txt}。")

if __name__ == '__main__':
    paths = [
        # '/home/shengguang/PycharmProjects/EurekDiffusion/vitmae_diffusion_working',
        # '/home/shengguang/PycharmProjects/EurekDiffusion/VideoMAEDiffusion',
        '/home/shengguang/PycharmProjects/OCTClassification',
        # '/home/shengguang/PycharmProjects/OCTClassification/app',
        # '/home/shengguang/PycharmProjects/OCTClassification/mae_pretrain'
    ]

    ignore_directories = [
        '/home/shengguang/PycharmProjects/OCTClassification/mae_pretrain',
        '/home/shengguang/PycharmProjects/OCTClassification/app/wandb',
        '/home/shengguang/PycharmProjects/OCTClassification/.venv',
        '/home/shengguang/PycharmProjects/FitDiT/preprocess',
        '/home/shengguang/PycharmProjects/FitDiT/local_model_dir',
        '/home/shengguang/PycharmProjects/OCTClassification/prompt'
        # 你可以在这里添加更多需要忽略的目录
    ]

    output_file = '/home/shengguang/PycharmProjects/OCTClassification/prompt/oct.txt'
    # output_file = '/home/diffusion/prompt/oct.txt'

    exporter = DirectoryCodeExporter(
        paths,
        output_file,
        ignore_dirs=ignore_directories,
        copy_to_clipboard=True  # 根据需要设置为 True 或 False
    )
    exporter.export_to_txt()
