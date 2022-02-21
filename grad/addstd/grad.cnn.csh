#!/bin/csh
# c-shell script for grad analysis.

# set lead month ��ǰ��
foreach LEAD(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23)

# set target season
@ TMON = 1
while ($TMON <= 12) # $ ��Ϊ������ǰ���������������滻��������һ�����������ݣ����磺echo $PATH��


setenv HHH '/mnt'                   # Main directory  set environment = setenv  ���û�������
#setenv RET 'retrain'				#��ѵ���ļ���

setenv MOD $LEAD'mon'$TMON					#��ѵ����ģ�� ��1mon1
setenv TYP 'cnngradstd'					#type


foreach conf ( 30 50 )      # Number of conv. features
foreach hidf ( 30 50 )      # Number of hidden neurons


setenv opname 'C'$conf'H'$hidf

echo $opname		#C30H30

# mkdir -p, -p�ǵݹ鴴�����Զ��������л������ڵĸ�(parent)Ŀ¼
mkdir -p $HHH/$TYP/$MOD                    #mnt/cnngrad/1mon1
mkdir -p $HHH/$TYP/$MOD/src

cd $HHH/$TYP/$MOD/src		    			
cp -f $HHH/$TYP/grad.cnn.sample .				#����Ҫ��ǰ��ѵ����������ļ�����

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

#����4��ģ�͵�ƽ���ݶ�
#���ڻ��� /mnt/cnngrad/1mon1/src/Ŀ¼��
cd $HHH/$TYP/$MOD/src
cp -f $HHH/$TYP/grad.CHmean.sample .			#����Ҫ��ǰ��ѵ���������retrain�ļ�����

sed "s/lmont/$MOD/g"                       grad.CHmean.sample > grad.CHmean.sample.py

python grad.CHmean.sample.py

@ TMON = $TMON + 1

end     #while TMON
end     #foreach LEAD
