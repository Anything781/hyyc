#!/bin/csh
# c-shell script for heatmap analysis.
setenv HHH '/mnt'                   # Main directory
#setenv RET 'retrain'				#��ѵ���ļ���

setenv MOD '2mon12'					#��ѵ����ģ�� 
setenv TYP 'cmip'					#type Ԥѵ��

setenv LEAD '2'                     # lead month
setenv TMON '12'                    # target month

foreach conf ( 30 50 )      # Number of conv. features
foreach hidf ( 30 50 )      # Number of hidden neurons


setenv opname 'C'$conf'H'$hidf

echo $opname		#C30H30

@ bei = 1
while ($bei <= 9)

# mkdir - p, -p�ǵݹ鴴�����Զ��������л������ڵĸ�(parent)Ŀ¼

mkdir -p $HHH/retrain/$MOD/$TYP/$opname
mkdir -p $HHH/retrain/$MOD/$TYP/$opname/0.$bei				#mnt/retrain/2mon12/cmip/C30H30/0.1
mkdir -p $HHH/retrain/$MOD/$TYP/src/$opname

cd $HHH/retrain/$MOD/$TYP/src/$opname
cp -f $HHH/retrain/2mon12retrain.cmip.sample .  #����Ҫ��ǰ��ѵ���������1mon12�ļ�����

# make layer
sed "s/chlist/$opname/g"						2mon12retrain.cmip.sample > tmp1
sed "s/lead_mon/$LEAD/g"                                tmp1 > tmp2
sed "s/target_mon/$TMON/g"                              tmp2 > tmp1
sed "s/convfilter/$conf/g"								tmp1 > tmp2         
sed "s/hiddfilter/$hidf/g"								tmp2 > tmp1
sed "s/lmont/$MOD/g"									tmp1 > tmp2
sed "s/multiple/$bei/g"                           tmp2 > 2mon12retrain.cmip.py

python 2mon12retrain.cmip.py

@ bei = $bei + 1

end   #while beishu

end    #foreach conf
end

