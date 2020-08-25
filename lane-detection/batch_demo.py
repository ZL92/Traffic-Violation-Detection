import os
import re
import demo
path = "/home/gym/video/" #文件夹目录
files= os.listdir(path) #得到文件夹下的所有文件名称
s = []
for file in files: #遍历文件夹
    if not os.path.isdir(path+file): #判断是否是文件夹，不是文件夹才打开
        #print("file {}".format(file))
        num = re.findall("\d+",str(file))[0]
        dirname = path+'img/'+num+'/'
        data = ''
        with open('configs/tusimple.py', 'r+') as f:
            for line in f.readlines():
                if(line.find('data_root') == 0):
                    #print(line)
                    line = "data_root = '" +dirname + "'\n"
                    #print(line)

                data += line
                print(line)

        with open('configs/tusimple.py', 'w') as f:
            f.writelines(data)



        os.system("python3 demo.py configs/tusimple.py")

