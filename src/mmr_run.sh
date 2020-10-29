#!/usr/bin/env bash




### see effects of co2
#for C2 in .2 .3 .4
#do
#    python pick_mmr.py -co1 .9 \
#                        -co2 $C2 \
#                        -co3 0.1 \
#                        -cos
#done


##########################
## see effects of co1 and co2  ##
#########################
#C1=.9
#C3=0.05
#C2=0.05
#echo "For $C1, $C2, $C3:"
#python pick_mmr.py -co1 $C1 \
#                        -co2 $C2 \
#                        -co3 $C3
#
#C1=.8
#C3=0.1
#C2=0.1
#echo "For $C1, $C2, $C3:"
#python pick_mmr.py -co1 $C1 \
#                        -co2 $C2 \
#                        -co3 $C3
#
#C1=.8
#C3=0.05
#C2=0.15
#echo "For $C1, $C2, $C3:"
#python pick_mmr.py -co1 $C1 \
#                        -co2 $C2 \
#                        -co3 $C3
#
#
#C1=.8
#C3=0.15
#C2=0.05
#echo "For $C1, $C2, $C3:"
#python pick_mmr.py -co1 $C1 \
#                        -co2 $C2 \
#                        -co3 $C3

##########################
## see effects of co2  ##
#########################
C1=.8
C3=0
C2=.2
echo "For $C1, $C2, $C3:"
python pick_mmr.py -co1 $C1 \
                        -co2 $C2 \
                        -co3 $C3

C1=.95
C3=0
C2=.05
echo "For $C1, $C2, $C3:"
python pick_mmr.py -co1 $C1 \
                        -co2 $C2 \
                        -co3 $C3

C1=.9
C3=0
C2=.1
echo "For $C1, $C2, $C3:"
python pick_mmr.py -co1 $C1 \
                        -co2 $C2 \
                        -co3 $C3


##########################
## see effects of co3  ##
#########################
#
#C1=.95
#C2=0
#C3=.05
#echo "For $C1, $C2, $C3:"
#python pick_mmr.py -co1 $C1 \
#                        -co2 $C2 \
#                        -co3 $C3
#
#
#C1=.9
#C2=0
#C3=.1
#echo "For $C1, $C2, $C3:"
#python pick_mmr.py -co1 $C1 \
#                        -co2 $C2 \
#                        -co3 $C3
#
#C1=.8
#C2=0
#C3=.2
#echo "For $C1, $C2, $C3:"
#python pick_mmr.py -co1 $C1 \
#                        -co2 $C2 \
#                        -co3 $C3
#
#
#C1=.7
#C2=0
#C3=.3
#echo "For $C1, $C2, $C3:"
#python pick_mmr.py -co1 $C1 \
#                        -co2 $C2 \
#                        -co3 $C3
#
#C1=0.6
#C2=0
#C3=0.4
#echo "For $C1, $C2, $C3:"
#python pick_mmr.py -co1 $C1 \
#                        -co2 $C2 \
#                        -co3 $C3