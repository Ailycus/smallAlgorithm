#!/bin/sh

echo "Test start...."

num=10
while [ "$num" != 200 ]
do
    python3 ./group_data.py $num >> min_group_log.log 2>&1
	num=$(($num +1))
done
echo "--over--"
