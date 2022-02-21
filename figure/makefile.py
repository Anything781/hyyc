import os

path = 'E:/hyyc/retrain0.2-0.7/allstd/' #最后加反斜杠 / 是为了以最后 / 之前目录为根目录，如果不加会以hyyc为根目录

for i in range(23):
  for j in range(12):
     isexists = os.path.exists(path+str(i+1)+'mon'+str(j+1))
     #isexists1 = os.path.exists(path+str(i+1)+'mon'+str(j+1)+'/splitmon')
     if not isexists:
        os.makedirs(path+str(i+1)+'mon'+str(j+1))
        print('new file:', path + str(i+1) + 'mon' + str(j+1))
     else:
       print('该split目录已存在')

'''     if not isexists1:
        os.makedirs(path+str(i+1)+'mon'+str(j+1)+'/splitmon')
        print('new file:', path + str(i+1) + 'mon' + str(j+1) + '/splitmon')
     else:
       print('该splitmon目录已存在')'''