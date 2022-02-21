#!/bin/csh
# c-shell script for heatmap analysis.
setenv HHH '/mnt'                   # Main directory
#setenv RET 'retrain'				#重训练文件夹

setenv MOD '2mon12'					#重训练的模型 
setenv TYP 'transfer'					#type 


foreach conf ( 30 50 )      # Number of conv. features
foreach hidf ( 30 50 )      # Number of hidden neurons


setenv opname 'C'$conf'H'$hidf

echo $opname		#C30H30


# mkdir - p, -p是递归创建，自动创建所有还不存在的父(parent)目录

cd $HHH/retrain/$MOD/$TYP/src/$opname				 #/mnt/retrain/2mon12/transfer/src/C30H30
cp -f $HHH/retrain/2mon12retrain.valid.sample .  #这里要提前把训练代码放在retrain文件夹下

# make layer
sed "s/chlist/$opname/g"						2mon12retrain.valid.sample > tmpv1
sed "s/convfilter/$conf/g"								tmpv1 > tmpv2         
sed "s/hiddfilter/$hidf/g"								tmpv2 > 2mon12retrain.valid.py
                           

python 2mon12retrain.valid.py

end    #foreach conf
end

#计算4种模型的平均重训练结果
cd $HHH/retrain/$MOD/$TYP/src/			                    # /mnt/retrain/2mon12/transfer/src
cp -f $HHH/retrain/2mon12retrain.validmean.sample .			#这里要提前把训练代码放在retrain文件夹下

sed "s/lmont/$MOD/g"                       2mon12retrain.validmean.sample > 2mon12retrain.validmean.sample.py

python 2mon12retrain.validmean.sample.py
echo $TYP  #放最后就会不打印，暂不知原因,
#后来找到原因是没有执行最后一行命令，在后边回车多个空白命令行，不让这行在最后一行就可以执行了