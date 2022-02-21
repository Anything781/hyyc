#!/bin/csh
# c-shell script for heatmap analysis.

# set lead month ��ǰ��
foreach LEAD(17 18 19 20 21 22 23)   #��ǰ��13-23

# set target season
@ TMON = 1
while ($TMON <= 12) # $ ��Ϊ������ǰ���������������滻��������һ�����������ݣ����磺echo $PATH��


setenv HHH '/mnt'						# Main directory
#setenv RET 'retrain'					#��ѵ���ļ���

setenv MOD $LEAD'mon'$TMON			#��ѵ����ģ��     �� 1mon1
setenv TYP 'cmip'						#type Ԥѵ��
setenv TYPE 'transfer'					#type Ǩ��ѵ��

echo $MOD								#ģ��

foreach conf ( 30 50 )      # Number of conv. features
foreach hidf ( 30 50 )      # Number of hidden neurons


setenv opname 'C'$conf'H'$hidf

echo $opname		#C30H30

@ bei = 1
while ($bei <= 9)

# mkdir - p, -p�ǵݹ鴴�����Զ��������л������ڵĸ�(parent)Ŀ¼
#����Ԥѵ���ļ���

mkdir -p $HHH/retrain/$MOD/$TYP/$opname
mkdir -p $HHH/retrain/$MOD/$TYP/$opname/0.$bei				#mnt/retrain/1mon1/cmip/C30H30/0.1
mkdir -p $HHH/retrain/$MOD/$TYP/src/$opname

#����Ǩ��ѵ���ļ���
mkdir -p $HHH/retrain/$MOD/$TYPE/$opname
mkdir -p $HHH/retrain/$MOD/$TYPE/$opname/0.$bei			#mnt/retrain/1mon1/transfer/C30H30/0.1
mkdir -p $HHH/retrain/$MOD/$TYPE/src/$opname

# ѵ�� cmip

cd $HHH/retrain/$MOD/$TYP/src/$opname						# 1mon1/cmip/src/c30h30
cp -f $HHH/retrain/retrain.cmip.sample .					#����Ҫ��ǰ��ѵ���������retrain�ļ�����

# make layer
sed "s/chlist/$opname/g"							retrain.cmip.sample > tmp1
sed "s/lead_mon/$LEAD/g"								 tmp1 > tmp2
sed "s/target_mon/$TMON/g"								 tmp2 > tmp1
sed "s/convfilter/$conf/g"								 tmp1 > tmp2         
sed "s/hiddfilter/$hidf/g"								 tmp2 > tmp1
sed "s/lmont/$MOD/g"									 tmp1 > tmp2
sed "s/multiple/$bei/g"								tmp2 > retrain.cmip.py

python retrain.cmip.py

# ѵ��transfer

cd $HHH/retrain/$MOD/$TYPE/src/$opname						# 1mon1/transfer/src/c30h30
cp -f $HHH/retrain/retrain.transfer.sample .				#����Ҫ��ǰ��ѵ���������retrain�ļ�����

# make layer
sed "s/chlist/$opname/g"							retrain.transfer.sample > tmpt1
sed "s/lead_mon/$LEAD/g"                                 tmpt1 > tmpt2
sed "s/target_mon/$TMON/g"                               tmpt2 > tmpt1
sed "s/convfilter/$conf/g"								 tmpt1 > tmpt2
sed "s/hiddfilter/$hidf/g"								 tmpt2 > tmpt1
sed "s/lmont/$MOD/g"									 tmpt1 > tmpt2
sed "s/multiple/$bei/g"								tmpt2 > retrain.transfer.py

python retrain.transfer.py

@ bei = $bei + 1

end   #while bei

# Ԥ�⣬�ǶԸ�����ṹ�±���c30h30������0.1-0.9��9��ģ�͵�Ԥ����

cd $HHH/retrain/$MOD/$TYPE/src/$opname						 # 1mon1/transfer/src/C30H30
cp -f $HHH/retrain/retrain.valid.sample .					 #����Ҫ��ǰ��ѵ���������retrain�ļ�����

# make layer
sed "s/chlist/$opname/g"							retrain.valid.sample > tmpv1
sed "s/lead_mon/$LEAD/g"                                 tmpv1 > tmpv2
sed "s/target_mon/$TMON/g"                               tmpv2 > tmpv1
sed "s/lmont/$MOD/g"									 tmpv1 > tmpv2
sed "s/convfilter/$conf/g"								 tmpv2 > tmpv1
sed "s/hiddfilter/$hidf/g"						    tmpv1 > retrain.valid.py

python retrain.valid.py


end    #foreach hidf
end	   #foreach conf

#�����ģ���� 4��ģ�͵�ƽ����ѵ�����
cd $HHH/retrain/$MOD/$TYPE/src								 # /mnt/retrain/1mon1/transfer/src
cp -f $HHH/retrain/retrain.validmean.sample .			 #����Ҫ��ǰ��ѵ���������retrain�ļ�����

sed "s/lmont/$MOD/g"                       retrain.validmean.sample > retrain.validmean.py

python retrain.validmean.py

@ TMON = $TMON + 1

end		# while TMON

end     #foreach LEAD
