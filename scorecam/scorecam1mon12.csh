#!/bin/csh
# c-shell script for heatmap analysis.

setenv HHH '/mnt'                   # Main directory

foreach conf ( 30 50 )      # Number of conv. features
foreach hidf ( 30 50 )      # Number of hidden neurons


setenv opname 'C'$conf'H'$hidf

echo $opname		#C30H30

mkdir -p $HHH/scorecam/1mon12
mkdir -p $HHH/scorecam/1mon12/src/$opname
mkdir -p $HHH/scorecam/1mon12/$opname

cd $HHH/scorecam/1mon12/src/$opname
cp -f $HHH/scorecam/scorecam1mon12.sample .

# make layer						
sed "s/convfilter/$conf/g"						scorecam1mon12.sample > tmp1
sed "s/hiddfilter/$hidf/g"                          tmp1 > tmp2
sed "s/chlist/$opname/g"                            tmp2 > scorecam1mon12.py


python scorecam1mon12.py

end
end

