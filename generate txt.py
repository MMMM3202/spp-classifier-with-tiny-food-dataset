import os


def generate(dir, label, datatype):
    files = os.listdir(dir)
    listText = open('Food_Classification_Dataset/label/' + datatype + '.txt', 'a')
    for file in files:
        file_path = os.path.join(dir, file)
        line = file_path + ' ' + label + '\n'
        listText.write(line)
    listText.close()


train_path = 'Food_Classification_Dataset\\train'
val_path = 'Food_Classification_Dataset\\val'
test_path = 'Food_Classification_Dataset\\test'


if __name__ == '__main__':
    labels = ['Bread', 'Hamburger', 'Kebab', 'Noodle', 'Rice']
    i = 0
    trainlist = os.listdir(train_path)
    vallist = os.listdir(val_path)
    testlist = os.listdir(test_path)
    for folder in trainlist:
        generate(os.path.join(train_path, folder), str(i), 'train')
        i += 1
    i = 0
    for folder in vallist:
        generate(os.path.join(val_path, folder), str(i), 'val')
        i += 1
    i = 0
    for folder in testlist:
        generate(os.path.join(test_path, folder), str(i), 'test')
        i += 1
