#!/bin/csh
# c-shell script for grad analysis.

# set lead month 提前期
foreach LEAD(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23)

# set target season
@ TMON = 1
while ($TMON <= 12) # $ 作为变量的前导符，用作变量替换，即引用一个变量的内容，比如：echo $PATH；


setenv HHH '/mnt'                   # Main directory  set environment = setenv  设置环境变量
#setenv RET 'retrain'				#重训练文件夹

setenv MOD $LEAD'mon'$TMON					#重训练的模型 如1mon1
setenv TYP 'cnngrad'					#type 


foreach conf ( 30 50 )      # Number of conv. features
foreach hidf ( 30 50 )      # Number of hidden neurons


setenv opname 'C'$conf'H'$hidf

echo $opname		#C30H30

# mkdir -p, -p是递归创建，自动创建所有还不存在的父(parent)目录
mkdir -p $HHH/$TYP/$MOD                    #mnt/cnngrad/1mon1
mkdir -p $HHH/$TYP/$MOD/src

cd $HHH/$TYP/$MOD/src		    			
cp -f $HHH/$TYP/grad.cnn.sample .				#这里要提前把训练代码放在文件夹下

# make layer
sed "s/chlist/$opname/g"						grad.cnn.sample > tmpg1
sed "s/lead_mon/$LEAD/g"                                tmpg1 > tmpg2
sed "s/target_mon/$TMON/g"                              tmpg2 > tmpg1
sed "s/lmont/$MOD/g"                                    tmpg1 > tmpg2
sed "s/convfilter/$conf/g"								tmpg2 > tmpg1   
sed "s/hiddfilter/$hidf/g"								tmpg1 > grad.cnn.sample.py
                           
python grad.cnn.sample.py

end    #foreach hidf
end	   #foreach conf

#计算4种模型的平均梯度
#现在还在 /mnt/cnngrad/1mon1/src/目录下
cd $HHH/$TYP/$MOD/src
cp -f $HHH/$TYP/grad.CHmean.sample .			#这里要提前把训练代码放在retrain文件夹下

sed "s/lmont/$MOD/g"                       grad.CHmean.sample > grad.CHmean.sample.py

python grad.CHmean.sample.py

@ TMON = $TMON + 1

end     #while TMON
end     #foreach LEAD
