import os

photos = os.listdir('data/train/images')

with open('data/train/labels.txt', 'w') as f:
    for photo in photos:
        name = photo.split('.')[0]
        if name == 'cat':
            f.write(name + '\n')
        else:
            f.write(name + '\n')
