#!/bin/csh		以csh shell来解释,表明这个脚本是用csh来解析的，因为各种shell的语法还是有细微差别的
# c-shell script(脚本) for transfer learning.用于迁移学习的c-shell脚本

# set lead month 提前期
foreach LEAD ( 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 )

# set target season
@ TMON = 1
while( $TMON <= 12 ) # $ 作为变量的前导符，用作变量替换，即引用一个变量的内容，比如：echo $PATH；
													#  set environment = setenv  设置环境变量
setenv CCC 'nino34_'$LEAD'month_'$TMON'_Transfer_20'        # case name (experiment name) 
setenv HHH '/home/jhkim/cnn/'                               # Main directory目录

foreach conf ( 30 50 )         # Number of conv. features
foreach hidf ( 30 50 )      # Number of hidden neurons 隐藏神经元数量

setenv GGG 0                       # GPU number (0~3)
setenv ENN 10                      # Ensemble number

setenv SMF 'soda.sst_t300.map.1871_1970.36mon.nc'         # Sample of training data
setenv LBF 'soda.nino.1873_1972.LAG.nc'                 # Label标签 of training data
setenv EVD 'godas.sst_t300.1980_2015.36mon.nc'         # Sample of evaluation评估 data
setenv EBD 'godas.nino.1982_2017_LAG.nc'       # Label of training data

setenv TTT 100                     # Total data size of training set 100年？
setenv SSS 100                     # Training data size of training set

setenv TEV 36                         # Total data size of  evaluation set验证集
setenv SEV 0                          # Starting point of evaluation set
setenv EEV 36                         # Evaluation size

setenv XXX 72                         # X dimension维
setenv YYY 24                         # Y dimension
setenv ZZZ  6                         # Z dimension

setenv BBB 20                        # Batch批次/量 size
setenv EEE 20                        # Epoch
setenv CDD 1.0                        # drop rate at the convolutional layer
setenv HDD 1.0                        # drop rate at the hidden layer

setenv opname 'C'$conf'H'$hidf		#C30H30....

# Number of ensemble
@ ens = 1
while ( $ens <= $ENN )

mkdir -p $HHH/output/$CCC		#$ 作为变量的前导符，用作变量替换，即引用一个变量的内容，比如：echo $PATH；
mkdir -p $HHH/output/$CCC/src	# mkdir - p ,- p是递归创建，自动创建所有还不存在的父(parent)目录
mkdir -p $HHH/output/$CCC/$opname		#创建目录$HHH、output、$CCC、$opname
mkdir -p $HHH/output/$CCC/$opname/EN$ens
mkdir -p $HHH/output/$CCC/$opname/ensmean

cd $HHH/output/$CCC/src  #进入目录....，cd = change directory
cp -f $HHH/sample/nino34.train_transfer.sample . #cp = copy file 
cp -f $HHH/sample/nino34.valid.sample .			# -f 强制覆盖已经存在的目标文件，不提示是否确认覆盖
cp -f $HHH/sample/nino34.ensmean.sample .

# Run Training
sed "s/convfilter/$conf/g"    nino34.train_transfer.sample > tmp1 #sed 's/要被取代的字串/新的字串/g'
sed "s/hiddfilter/$hidf/g"                            tmp1 > tmp2
sed "s/samfile/$SMF/g"                                tmp2 > tmp1
sed "s/labfile/$LBF/g"                                tmp1 > tmp2
sed "s/opfname/$opname/g"                             tmp2 > tmp1
sed "s/case/$CCC/g"                                   tmp1 > tmp2
sed "s/batsiz/$BBB/g"                                 tmp2 > tmp1
sed "s/xxx/$XXX/g"                                    tmp1 > tmp2
sed "s/yyy/$YYY/g"                                    tmp2 > tmp1
sed "s/epoch/$EEE/g"                                  tmp1 > tmp2
sed "s/TOTSIZ/$TTT/g"                                 tmp2 > tmp1
sed "s/SAMSIZ/$SSS/g"                                 tmp1 > tmp2
sed "s/CDRP/$CDD/g"                                   tmp2 > tmp1
sed "s/HDRP/$HDD/g"                                   tmp1 > tmp2
sed "s/zzz/$ZZZ/g"                                    tmp2 > tmp1
sed "s/lead_mon/$LEAD/g"                              tmp1 > tmp2
sed "s/target_mon/$TMON/g"                            tmp2 > tmp1
sed "s/number_gpu/$GGG/g"                             tmp1 > tmp2
sed "s#home_directory#$HHH#g"                         tmp2 > tmp1
sed "s/member/$ens/g"                                 tmp1 > nino34.train_transfer.py

python nino34.train_transfer.py

# Run Evaluation
sed "s/convfilter/$conf/g"          nino34.valid.sample > tmp1
sed "s/hiddfilter/$hidf/g"                         tmp1 > tmp2
sed "s/samfile/$EVD/g"                             tmp2 > tmp1
sed "s/opfname/$opname/g"                          tmp1 > tmp2
sed "s/case/$CCC/g"                                tmp2 > tmp1
sed "s/xxx/$XXX/g"                                 tmp1 > tmp2
sed "s/yyy/$YYY/g"                                 tmp2 > tmp1
sed "s/TOTSIZ/$TEV/g"                              tmp1 > tmp2
sed "s/SAMSIZ/$SEV/g"                              tmp2 > tmp1
sed "s/TSTSIZ/$EEV/g"                              tmp1 > tmp2
sed "s/zzz/$ZZZ/g"                                 tmp2 > tmp1
sed "s/lead_mon/$LEAD/g"                           tmp1 > tmp2
sed "s/target_mon/$TMON/g"                         tmp2 > tmp1
sed "s/number_gpu/$GGG/g"                          tmp1 > tmp2
sed "s#home_directory#$HHH#g"                      tmp2 > tmp1
sed "s/member/$ens/g"                              tmp1 > nino34.valid.py

python nino34.valid.py

@ ens = $ens + 1
end

#compute ensemble mean
sed "s/opfname/$opname/g"    nino34.ensmean.sample > tmp1
sed "s/case/$CCC/g"                           tmp1 > tmp2
sed "s/TSTSIZ/$EEV/g"                         tmp2 > tmp1
sed "s/convfilter/$conf/g"                    tmp1 > tmp2
sed "s#home_directory#$HHH#g"                 tmp2 > tmp1
sed "s/numen/$ENN/g"                          tmp1 > nino34.ensmean.py

python nino34.ensmean.py


end
end

@ TMON = $TMON + 1
end

end

