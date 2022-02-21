#!/bin/csh
# c-shell script for heatmap analysis.
setenv HHH '/mnt'                   # Main directory
#setenv RET 'retrain'				#��ѵ���ļ���

setenv MOD '2mon12'					#��ѵ����ģ�� 
setenv TYP 'grad'					#type 

setenv LEAD '2'                     # lead month
setenv TMON '12'                    # target month

foreach conf ( 30 50 )      # Number of conv. features
foreach hidf ( 30 50 )      # Number of hidden neurons


setenv opname 'C'$conf'H'$hidf

echo $opname		#C30H30


# mkdir -p, -p�ǵݹ鴴�����Զ��������л������ڵĸ�(parent)Ŀ¼
mkdir -p $HHH/retrain/$MOD/$TYP                    #mnt/retrain/2mon12/grad�����ݶȽ��������grad�ļ����¼���
		
mkdir -p $HHH/retrain/$MOD/$TYP/src

cd $HHH/retrain/$MOD/$TYP/src		    			#/mnt/retrain/1mon12/transfer/src
cp -f $HHH/retrain/grad.2mon12.sample .				#����Ҫ��ǰ��ѵ���������retrain�ļ�����

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

#����4��ģ�͵�ƽ���ݶ�
#���ڻ��� /mnt/ retrain/1mon12/transfer/src/Ŀ¼��
cp -f $HHH/retrain/grad.CHmean.sample .			#����Ҫ��ǰ��ѵ���������retrain�ļ�����

sed "s/lmont/$MOD/g"                       grad.CHmean.sample > grad.CHmean.sample.py

python grad.CHmean.sample.py

