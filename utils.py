from config import *
import os


def make_new_dir():
    checkpath(SAVE_PATH)
    checkpath(SAVE_PATH + "/model")
    checkpath(SAVE_PATH + "/log")
    checkpath(SAVE_PATH + "/visualize")
    print(SAVE_PATH)

def checkpath(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def load_weights(model, weights_path):
    if weights_path:
        model_dict = model.state_dict()
        pretrained_dict = t.load(weights_path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print("load weights [{}].\n".format(weights_path))
    else:
        print("not load weights.\n")
    return model

def write_csv(results,file_name):
    import csv
    with open(file_name,'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id','label'])
        writer.writerows(results)