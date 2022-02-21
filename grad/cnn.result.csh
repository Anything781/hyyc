#!/bin/csh
# c-shell script for heatmap analysis.

# set lead month 提前期
foreach LEAD(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23)

# set target season
@ TMON = 1
while ($TMON <= 12) # $ 作为变量的前导符，用作变量替换，即引用一个变量的内容，比如：echo $PATH；

setenv CCC 'nino34_'$LEAD'month_'$TMON'_Transfer_20'
setenv HHH '/mnt'                                          # Main directory
setenv TYP 'cnnresult'					                   # type 

setenv MOD $LEAD'mon'$TMON					        #求原CNN预测结果的模型 

echo $MOD		#  1month1


# mkdir -p, -p是递归创建，自动创建所有还不存在的父(parent)目录
mkdir -p $HHH/$TYP/src
cd $HHH/$TYP/src			 #/mnt/retrain/1mon12/transfer/src/C30H30
cp -f $HHH/retrain/cnn.CHmean.sample .  #这里要提前把训练代码放在retrain文件夹下

# make layer

sed "s/model/$CCC/g"					cnn.CHmean.sample > tmpc1
sed "s/lmont/$MOD/g"				    tmpc1 > cnn.CHmean.sample.py
								                       
python cnn.CHmean.sample.py

@ TMON = $TMON + 1

end    #while TMON
end    #foreach LEAD

