#!/bin/csh		��csh shell������,��������ű�����csh�������ģ���Ϊ����shell���﷨������ϸ΢����
# c-shell script(�ű�) for transfer learning.����Ǩ��ѧϰ��c-shell�ű�

# set lead month ��ǰ��
foreach LEAD ( 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 )

# set target season
@ TMON = 1
while( $TMON <= 12 ) # $ ��Ϊ������ǰ���������������滻��������һ�����������ݣ����磺echo $PATH��
													#  set environment = setenv  ���û�������
setenv CCC 'nino34_'$LEAD'month_'$TMON'_Transfer_20'        # case name (experiment name) 
setenv HHH '/home/jhkim/cnn/'                               # Main directoryĿ¼

foreach conf ( 30 50 )         # Number of conv. features
foreach hidf ( 30 50 )      # Number of hidden neurons ������Ԫ����

setenv GGG 0                       # GPU number (0~3)
setenv ENN 10                      # Ensemble number

setenv SMF 'soda.sst_t300.map.1871_1970.36mon.nc'         # Sample of training data
setenv LBF 'soda.nino.1873_1972.LAG.nc'                 # Label��ǩ of training data
setenv EVD 'godas.sst_t300.1980_2015.36mon.nc'         # Sample of evaluation���� data
setenv EBD 'godas.nino.1982_2017_LAG.nc'       # Label of training data

setenv TTT 100                     # Total data size of training set 100�ꣿ
setenv SSS 100                     # Training data size of training set

setenv TEV 36                         # Total data size of  evaluation set��֤��
setenv SEV 0                          # Starting point of evaluation set
setenv EEV 36                         # Evaluation size

setenv XXX 72                         # X dimensionά
setenv YYY 24                         # Y dimension
setenv ZZZ  6                         # Z dimension

setenv BBB 20                        # Batch����/�� size
setenv EEE 20                        # Epoch
setenv CDD 1.0                        # drop rate at the convolutional layer
setenv HDD 1.0                        # drop rate at the hidden layer

setenv opname 'C'$conf'H'$hidf		#C30H30....

# Number of ensemble
@ ens = 1
while ( $ens <= $ENN )

mkdir -p $HHH/output/$CCC		#$ ��Ϊ������ǰ���������������滻��������һ�����������ݣ����磺echo $PATH��
mkdir -p $HHH/output/$CCC/src	# mkdir - p ,- p�ǵݹ鴴�����Զ��������л������ڵĸ�(parent)Ŀ¼
mkdir -p $HHH/output/$CCC/$opname		#����Ŀ¼$HHH��output��$CCC��$opname
mkdir -p $HHH/output/$CCC/$opname/EN$ens
mkdir -p $HHH/output/$CCC/$opname/ensmean

cd $HHH/output/$CCC/src  #����Ŀ¼....��cd = change directory
cp -f $HHH/sample/nino34.train_transfer.sample . #cp = copy file 
cp -f $HHH/sample/nino34.valid.sample .			# -f ǿ�Ƹ����Ѿ����ڵ�Ŀ���ļ�������ʾ�Ƿ�ȷ�ϸ���
cp -f $HHH/sample/nino34.ensmean.sample .

# Run Training
sed "s/convfilter/$conf/g"    nino34.train_transfer.sample > tmp1 #sed 's/Ҫ��ȡ�����ִ�/�µ��ִ�/g'
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

