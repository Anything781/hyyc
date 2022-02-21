#!/bin/csh
# c-shell script for heatmap analysis.

setenv HHH '/mnt'                   # Main directory
setenv SIZ '2x2'

foreach conf ( 30 50 )      # Number of conv. features
foreach hidf ( 30 50 )      # Number of hidden neurons

setenv ENN 10                      # Ensemble number

setenv opname 'C'$conf'H'$hidf

echo $opname		#C30H30

# Number of ensemble
@ ens = 1
while ($ens <= $ENN)


mkdir -p $HHH/disturbinput/1month12to0/$SIZ
mkdir -p $HHH/disturbinput/1month12to0/$SIZ/src/$opname
mkdir -p $HHH/disturbinput/1month12to0/$SIZ/$opname

cd $HHH/disturbinput/1month12to0/$SIZ/src/$opname
cp -f $HHH/disturbinput/2x2disturbto0.1mon12.sample .

# make layer
sed "s/member/$ens/g"					  2x2disturbto0.1mon12.sample > tmp1
sed "s/convfilter/$conf/g"							tmp1 > tmp2
sed "s/hiddfilter/$hidf/g"                          tmp2 > tmp1
sed "s/chlist/$opname/g"                            tmp1 > 2x2disturbto0.1mon12.py


python 2x2disturbto0.1mon12.py

@ ens = $ens + 1
end						#while ens

end
end

