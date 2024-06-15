import os
import argparse
def get_file_list(path):
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list

def path_to_file_list(path):
    lines = open(path, 'r').read().split('\n')
    return lines

def file_list_to_json(file_list):
    template_start = '{\"input\":\"'
    template_end = '\"}'
    processed_file_list = []
    for file in file_list:
        if '\\' in file:
            file = file.replace('\\', '\\\\')
        if '/' or '"' in file:
            file = file.replace('/', '\\/')
            file = file.replace('"', '\\"')

        processed_file_list.append(template_start + file + template_end)
    return processed_file_list

def train_file_list_to_json(informal_file_list, formal_file_list):
    # Preprocess unwanted characters
    def process_file(file):
        if '\\' in file:
            file = file.replace('\\', '\\\\')
        if '/' or '"' in file:
            file = file.replace('/', '\\/')
            file = file.replace('"', '\\"')
        return file

    # Template for json file
    template_start = '{\"en\":\"'
    template_mid = '\",\"du\":\"'
    template_end = '\"}'

    processed_file_list = []
    for informal_file, formal_file in zip(informal_file_list, formal_file_list):
        informal_file = process_file(informal_file)
        formal_file = process_file(formal_file)

        processed_file_list.append(template_start + informal_file + template_mid + formal_file + template_end)
    return processed_file_list

def write_file_list(file_list, path):
    with open(path, 'w') as f:
        for file in file_list:
            f.write(file + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default='./GYAFC_Corpus/GYAFC_Corpus/Family_Relationships/test/')
    parser.add_argument("--en_path", default='./GYAFC_Corpus/GYAFC_Corpus/Family_Relationships/test/informal.txt')
    parser.add_argument("--du_path", default='./GYAFC_Corpus/GYAFC_Corpus/Family_Relationships/test/formal.txt')

    args =  parser.parse_args()

    # this is the path to the folder containing the parallel corpus
    path = args.path
    # this is the path to the folder containing the english sentences
    en_path = args.en_path
    # this is the path to the folder containing the german sentences
    du_path = args.du_path

    en_file_list = path_to_file_list(en_path)
    du_file_list = path_to_file_list(du_path)

    processed_file_list = train_file_list_to_json(en_file_list, du_file_list)

    write_file_list(processed_file_list, path+'test.json')