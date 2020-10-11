import os
import re
import demo

path = "./challenge_testing_data/testing_data/video1/"  # 文件夹目录
files = os.listdir(path)
try:
    files.remove('img')
except:
    pass
files.sort(key=lambda x: int(x[:-4]))
s = 0


for file in files:  # 遍历文件夹
    # s+=1
    print('In batch_demo.py, file is {}'.format(file))
    # print('This is file{}'.format(s))
    if not os.path.isdir(path + file):  # 判断是否是文件夹，不是文件夹才打开
        # print("path+file is {}".format(path+file))
        num = re.findall("\d+", str(file))[0]
        dirname = path + 'img/' + num + '/'
        data = ''
        with open('configs/tusimple.py', 'r+') as f:
            for line in f.readlines():
                if (line.find('data_root') == 0):
                    # print(line)
                    line = "data_root = '" + dirname + "'\n"
                    # print(line)

                data += line
                # print(line)

        with open('configs/tusimple.py', 'w') as f:
            f.writelines(data)

        os.system("python3 demo.py configs/tusimple.py --test_model tusimple_18.pth")