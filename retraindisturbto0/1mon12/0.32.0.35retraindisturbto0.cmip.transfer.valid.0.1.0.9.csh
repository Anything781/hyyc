#!/bin/csh
# c-shell script for heatmap analysis.


setenv HHH '/mnt'						# Main directory
setenv RET 'retraindisturbto0'					#��ѵ���ļ���

setenv MOD 're1mon12'			#��ѵ����ģ��     �� 1mon1
setenv TYP 'cmip'						#type Ԥѵ��
setenv TYPE 'transfer'					#type Ǩ��ѵ��

echo $MOD								#ģ��

foreach conf ( 30 50 )      # Number of conv. features
foreach hidf ( 30 50 )      # Number of hidden neurons


setenv opname 'C'$conf'H'$hidf

echo $opname		#C30H30

@ bei = 32
while ($bei <= 35 )									#��ֹ���С��0.5���־������⣬��Ϊ��50��cmip��transfer��0.1��Ϊ0.001����

# mkdir - p, -p�ǵݹ鴴�����Զ��������л������ڵĸ�(parent)Ŀ¼

#����Ԥѵ���ļ���
mkdir -p $HHH/$RET/$MOD/$TYP/$opname
mkdir -p $HHH/$RET/$MOD/$TYP/$opname/0.$bei			#mnt/retraindisturbto0/1mon12/cmip/C30H30/0.1
mkdir -p $HHH/$RET/$MOD/$TYP/src/$opname

#����Ǩ��ѵ���ļ���
mkdir -p $HHH/$RET/$MOD/$TYPE/$opname
mkdir -p $HHH/$RET/$MOD/$TYPE/$opname/0.$bei			#mnt/retraindisturbto0/1mon12/transfer/C30H30/0.1
mkdir -p $HHH/$RET/$MOD/$TYPE/src/$opname

# ѵ�� cmip

cd $HHH/$RET/$MOD/$TYP/src/$opname						# 1mon12/cmip/src/c30h30
cp -f $HHH/$RET/$MOD/retrain.cmip.sample .					#����Ҫ��ǰ��ѵ���������retraindisturbto0�ļ�����

# make layer
sed "s/chlist/$opname/g"							retrain.cmip.sample > tmp1
sed "s/convfilter/$conf/g"								 tmp1 > tmp2         
sed "s/hiddfilter/$hidf/g"								 tmp2 > tmp1
sed "s/lmont/$MOD/g"									 tmp1 > tmp2
sed "s/document/$RET/g"									 tmp2 > tmp1
sed "s/multiple/$bei/g"								tmp1 > retrain.cmip.py

python retrain.cmip.py

# ѵ��transfer

cd $HHH/$RET/$MOD/$TYPE/src/$opname						# 1mon12/transfer/src/c30h30
cp -f $HHH/$RET/$MOD/retrain.transfer.sample .				#����Ҫ��ǰ��ѵ���������retraindisturbto0�ļ�����

# make layer
sed "s/chlist/$opname/g"							retrain.transfer.sample > tmpt1
sed "s/convfilter/$conf/g"								 tmpt1 > tmpt2
sed "s/hiddfilter/$hidf/g"								 tmpt2 > tmpt1
sed "s/lmont/$MOD/g"									 tmpt1 > tmpt2
sed "s/document/$RET/g"									 tmpt2 > tmpt1
sed "s/multiple/$bei/g"								tmpt1 > retrain.transfer.py

python retrain.transfer.py

@ bei = $bei + 1									#��0.1����������

end   #while bei

# Ԥ�⣬�ǶԸ�����ṹ�±���c30h30������0.1-0.9��9��ģ�͵�Ԥ����

cd $HHH/$RET/$MOD/$TYPE/src/$opname					 # 1mon1/transfer/src/C30H30
cp -f $HHH/$RET/$MOD/retrain.valid.sample .				 #����Ҫ��ǰ��ѵ���������retraindisturbto0�ļ�����

# make layer
sed "s/chlist/$opname/g"							retrain.valid.sample > tmpv1
sed "s/lmont/$MOD/g"									 tmpv1 > tmpv2
sed "s/convfilter/$conf/g"								 tmpv2 > tmpv1
sed "s/document/$RET/g"									 tmpv1 > tmpv2
sed "s/hiddfilter/$hidf/g"						    tmpv2 > retrain.valid.py

python retrain.valid.py


end    #foreach hidf
end	   #foreach conf

#�����ģ���� 4��ģ�͵�ƽ����ѵ�����
cd $HHH/$RET/$MOD/$TYPE/src							 # 1mon1/transfer/src
cp -f $HHH/$RET/$MOD/retrain.validmean.sample .			 #����Ҫ��ǰ��ѵ���������retraindisturbto0�ļ�����

sed "s/lmont/$MOD/g"								 retrain.validmean.sample > retrain.validmean.py

python retrain.validmean.py
