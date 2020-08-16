import yaml
import shutil
import os

def load_yml(f_name):
    with open(f_name, 'r', encoding="utf-8") as f:
        res = yaml.full_load(f)
    return res

def save_yml(f_name, data):
    with open(f_name, 'w', encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False,\
                          encoding = 'utf-8', \
                          allow_unicode = True)

#--------------dir---------------------

def create_dir(dir_name):

    if not os.path.exists(dir_name):
        try:
            os.makedirs(dir_name)
        except OSError as exc:
            raise

def remove_dir(dir_name):
    shutil.rmtree(dir_name, True)


def clear_dir(dirpath):
    '''清空文件夹,但保留文件夹存在'''
    shutil.rmtree(dirpath)
    os.mkdir(dirpath)
