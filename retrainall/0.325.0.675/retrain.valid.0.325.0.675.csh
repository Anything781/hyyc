#!/bin/csh
# c-shell script for heatmap analysis.

# set lead month ��ǰ��
foreach LEAD(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23)  #��ǰ�� 

# set target season
@ TMON = 1
while ($TMON <= 12) # $ ��Ϊ������ǰ���������������滻��������һ�����������ݣ����磺echo $PATH��


setenv HHH '/mnt'						# Main directory
setenv RET 'retrain0.325.0.675'					#��ѵ���ļ���

setenv MOD $LEAD'mon'$TMON			#��ѵ����ģ��     �� 1mon1
setenv TYP 'cmip'						#type Ԥѵ��
setenv TYPE 'transfer'					#type Ǩ��ѵ��

echo $MOD								#ģ��

foreach conf ( 30 50 )      # Number of conv. features
foreach hidf ( 30 50 )      # Number of hidden neurons


setenv opname 'C'$conf'H'$hidf

echo $opname		#C30H30


# Ԥ�⣬�ǶԸ�����ṹ�±���c30h30������0.325-0.675��0.05������8��ģ�͵�Ԥ����

cd $HHH/$RET/$MOD/$TYPE/src/$opname					 # 1mon1/transfer/src/C30H30
cp -f $HHH/$RET/retrain.valid.sample .				 #����Ҫ��ǰ��ѵ���������retrain0.2.0.7�ļ�����

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

#�����ģ���� 4��ģ�͵�ƽ����ѵ�����
cd $HHH/$RET/$MOD/$TYPE/src							 # 1mon1/transfer/src
cp -f $HHH/$RET/retrain.validmean.sample .			 #����Ҫ��ǰ��ѵ���������retrain0.2.0.7�ļ�����

sed "s/lmont/$MOD/g"								 retrain.validmean.sample > tmpm1
sed "s/document/$RET/g"  							 tmpm1 > retrain.validmean.py

python retrain.validmean.py

@ TMON = $TMON + 1

end		# while TMON

end     #foreach LEAD
