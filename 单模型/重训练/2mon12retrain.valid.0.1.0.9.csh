#!/bin/csh
# c-shell script for heatmap analysis.
setenv HHH '/mnt'                   # Main directory
#setenv RET 'retrain'				#��ѵ���ļ���

setenv MOD '2mon12'					#��ѵ����ģ�� 
setenv TYP 'transfer'					#type 


foreach conf ( 30 50 )      # Number of conv. features
foreach hidf ( 30 50 )      # Number of hidden neurons


setenv opname 'C'$conf'H'$hidf

echo $opname		#C30H30


# mkdir - p, -p�ǵݹ鴴�����Զ��������л������ڵĸ�(parent)Ŀ¼

cd $HHH/retrain/$MOD/$TYP/src/$opname				 #/mnt/retrain/2mon12/transfer/src/C30H30
cp -f $HHH/retrain/2mon12retrain.valid.sample .  #����Ҫ��ǰ��ѵ���������retrain�ļ�����

# make layer
sed "s/chlist/$opname/g"						2mon12retrain.valid.sample > tmpv1
sed "s/convfilter/$conf/g"								tmpv1 > tmpv2         
sed "s/hiddfilter/$hidf/g"								tmpv2 > 2mon12retrain.valid.py
                           

python 2mon12retrain.valid.py

end    #foreach conf
end

#����4��ģ�͵�ƽ����ѵ�����
cd $HHH/retrain/$MOD/$TYP/src/			                    # /mnt/retrain/2mon12/transfer/src
cp -f $HHH/retrain/2mon12retrain.validmean.sample .			#����Ҫ��ǰ��ѵ���������retrain�ļ�����

sed "s/lmont/$MOD/g"                       2mon12retrain.validmean.sample > 2mon12retrain.validmean.sample.py

python 2mon12retrain.validmean.sample.py
echo $TYP  #�����ͻ᲻��ӡ���ݲ�֪ԭ��,
#�����ҵ�ԭ����û��ִ�����һ������ں�߻س�����հ������У��������������һ�оͿ���ִ����