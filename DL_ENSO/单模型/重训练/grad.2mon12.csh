#!/bin/csh
# c-shell script for heatmap analysis.
setenv HHH '/mnt'                   # Main directory
#setenv RET 'retrain'				#重训练文件夹

setenv MOD '2mon12'					#重训练的模型 
setenv TYP 'grad'					#type 

setenv LEAD '2'                     # lead month
setenv TMON '12'                    # target month

foreach conf ( 30 50 )      # Number of conv. features
foreach hidf ( 30 50 )      # Number of hidden neurons


setenv opname 'C'$conf'H'$hidf

echo $opname		#C30H30


# mkdir -p, -p是递归创建，自动创建所有还不存在的父(parent)目录
mkdir -p $HHH/retrain/$MOD/$TYP                    #mnt/retrain/2mon12/grad，把梯度结果都放在grad文件夹下即可
		
mkdir -p $HHH/retrain/$MOD/$TYP/src

cd $HHH/retrain/$MOD/$TYP/src		    			#/mnt/retrain/1mon12/transfer/src
cp -f $HHH/retrain/grad.2mon12.sample .				#这里要提前把训练代码放在retrain文件夹下

# make layer
sed "s/chlist/$opname/g"						grad.2mon12.sample > tmpg1
sed "s/lead_mon/$LEAD/g"                                tmpg1 > tmpg2
sed "s/target_mon/$TMON/g"                              tmpg2 > tmpg1
sed "s/lmont/$MOD/g"                                    tmpg1 > tmpg2
sed "s/convfilter/$conf/g"								tmpg2 > tmpg1   
sed "s/hiddfilter/$hidf/g"								tmpg1 > grad.2mon12.sample.py
                           
python grad.2mon12.sample.py

end    #foreach conf
end

#计算4种模型的平均梯度
#现在还在 /mnt/ retrain/1mon12/transfer/src/目录下
cp -f $HHH/retrain/grad.CHmean.sample .			#这里要提前把训练代码放在retrain文件夹下

sed "s/lmont/$MOD/g"                       grad.CHmean.sample > grad.CHmean.sample.py

python grad.CHmean.sample.py

