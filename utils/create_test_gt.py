import os
import shutil

from utils import preprocessor

if __name__ == "__main__":
    os.chdir("..")
    _, _, test_dataset = preprocessor.get_datasets(task=1)
    files = [test_dataset.get_name(i) for i in range(len(test_dataset))]
    for file_name in files:
        path1 = os.path.join("data/labels/", file_name[:-4]+"_clothes.png")
        path2 = os.path.join("data/labels/", file_name[:-4]+"_person.png")

        if os.path.exists(path1):
            shutil.copy(path1, os.path.join("data/test", file_name[:-4]+"_clothes.png"))
        if os.path.exists(path2):
            shutil.copy(path2, os.path.join("data/test", file_name[:-4]+"_person.png"))