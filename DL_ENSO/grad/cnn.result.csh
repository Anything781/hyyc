#!/bin/csh
# c-shell script for heatmap analysis.

# set lead month ��ǰ��
foreach LEAD(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23)

# set target season
@ TMON = 1
while ($TMON <= 12) # $ ��Ϊ������ǰ���������������滻��������һ�����������ݣ����磺echo $PATH��

setenv CCC 'nino34_'$LEAD'month_'$TMON'_Transfer_20'
setenv HHH '/mnt'                                          # Main directory
setenv TYP 'cnnresult'					                   # type 

setenv MOD $LEAD'mon'$TMON					        #��ԭCNNԤ������ģ�� 

echo $MOD		#  1month1


# mkdir -p, -p�ǵݹ鴴�����Զ��������л������ڵĸ�(parent)Ŀ¼
mkdir -p $HHH/$TYP/src
cd $HHH/$TYP/src			 #/mnt/retrain/1mon12/transfer/src/C30H30
cp -f $HHH/retrain/cnn.CHmean.sample .  #����Ҫ��ǰ��ѵ���������retrain�ļ�����

# make layer

sed "s/model/$CCC/g"					cnn.CHmean.sample > tmpc1
sed "s/lmont/$MOD/g"				    tmpc1 > cnn.CHmean.sample.py
								                       
python cnn.CHmean.sample.py

@ TMON = $TMON + 1

end    #while TMON
end    #foreach LEAD

