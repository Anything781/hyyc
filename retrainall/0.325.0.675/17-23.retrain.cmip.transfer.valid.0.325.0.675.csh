#!/bin/csh
# c-shell script for heatmap analysis.

# set lead month 提前期
foreach LEAD(17 18 19 20 21 22 23)  #提前期 17-23

# set target season
@ TMON = 1
while ($TMON <= 12) # $ 作为变量的前导符，用作变量替换，即引用一个变量的内容，比如：echo $PATH；


setenv HHH '/mnt'						# Main directory
setenv RET 'retrain0.325.0.675'					#重训练文件夹

setenv MOD $LEAD'mon'$TMON			#重训练的模型     如 1mon1
setenv TYP 'cmip'						#type 预训练
setenv TYPE 'transfer'					#type 迁移训练

echo $MOD								#模型

foreach conf ( 30 50 )      # Number of conv. features
foreach hidf ( 30 50 )      # Number of hidden neurons


setenv opname 'C'$conf'H'$hidf

echo $opname		#C30H30

@ bei = 325
while ($bei <= 675)									#防止因加小数0.5出现精度问题，改为加50，cmip和transfer乘0.1改为0.001即可

# mkdir - p, -p是递归创建，自动创建所有还不存在的父(parent)目录
#创建预训练文件夹

mkdir -p $HHH/$RET/$MOD/$TYP/$opname
mkdir -p $HHH/$RET/$MOD/$TYP/$opname/0.$bei			#mnt/retrain0.2.0.7/1mon1/cmip/C30H30/0.1
mkdir -p $HHH/$RET/$MOD/$TYP/src/$opname

#创建迁移训练文件夹
mkdir -p $HHH/$RET/$MOD/$TYPE/$opname
mkdir -p $HHH/$RET/$MOD/$TYPE/$opname/0.$bei			#mnt/retrain0.2.0.7/1mon1/transfer/C30H30/0.1
mkdir -p $HHH/$RET/$MOD/$TYPE/src/$opname

# 训练 cmip

cd $HHH/$RET/$MOD/$TYP/src/$opname						# 1mon1/cmip/src/c30h30
cp -f $HHH/$RET/retrain.cmip.sample .					#这里要提前把训练代码放在retrain0.2.0.7文件夹下

# make layer
sed "s/chlist/$opname/g"							retrain.cmip.sample > tmp1
sed "s/lead_mon/$LEAD/g"								 tmp1 > tmp2
sed "s/target_mon/$TMON/g"								 tmp2 > tmp1
sed "s/convfilter/$conf/g"								 tmp1 > tmp2         
sed "s/hiddfilter/$hidf/g"								 tmp2 > tmp1
sed "s/lmont/$MOD/g"									 tmp1 > tmp2
sed "s/document/$RET/g"									 tmp2 > tmp1
sed "s/multiple/$bei/g"								tmp1 > retrain.cmip.py

python retrain.cmip.py

# 训练transfer

cd $HHH/$RET/$MOD/$TYPE/src/$opname						# 1mon1/transfer/src/c30h30
cp -f $HHH/$RET/retrain.transfer.sample .				#这里要提前把训练代码放在retrain0.2.0.7文件夹下

# make layer
sed "s/chlist/$opname/g"							retrain.transfer.sample > tmpt1
sed "s/lead_mon/$LEAD/g"                                 tmpt1 > tmpt2
sed "s/target_mon/$TMON/g"                               tmpt2 > tmpt1
sed "s/convfilter/$conf/g"								 tmpt1 > tmpt2
sed "s/hiddfilter/$hidf/g"								 tmpt2 > tmpt1
sed "s/lmont/$MOD/g"									 tmpt1 > tmpt2
sed "s/document/$RET/g"									 tmpt2 > tmpt1
sed "s/multiple/$bei/g"								tmpt1 > retrain.transfer.py

python retrain.transfer.py

@ bei = $bei + 50									#按0.5倍步长增加

end   #while bei

# 预测，是对该网络结构下比如c30h30，所有0.325-0.675共0.05步长共8个模型的预测结果

cd $HHH/$RET/$MOD/$TYPE/src/$opname					 # 1mon1/transfer/src/C30H30
cp -f $HHH/$RET/retrain.valid.sample .				 #这里要提前把训练代码放在retrain0.2.0.7文件夹下

# make layer
sed "s/chlist/$opname/g"							retrain.valid.sample > tmpv1
sed "s/lead_mon/$LEAD/g"                                 tmpv1 > tmpv2
sed "s/target_mon/$TMON/g"                               tmpv2 > tmpv1
sed "s/lmont/$MOD/g"									 tmpv1 > tmpv2
sed "s/convfilter/$conf/g"								 tmpv2 > tmpv1
sed "s/document/$RET/g"									 tmpv1 > tmpv2
sed "s/hiddfilter/$hidf/g"						    tmpv2 > retrain.valid.py

python retrain.valid.py


end    #foreach hidf
end	   #foreach conf

#计算该模型下 4种模型的平均重训练结果
cd $HHH/$RET/$MOD/$TYPE/src							 # 1mon1/transfer/src
cp -f $HHH/$RET/retrain.validmean.sample .			 #这里要提前把训练代码放在retrain0.2.0.7文件夹下

sed "s/lmont/$MOD/g"								 retrain.validmean.sample > tmpm1
sed "s/document/$RET/g"  							 tmpm1 > retrain.validmean.py

python retrain.validmean.py

@ TMON = $TMON + 1

end		# while TMON

end     #foreach LEAD
