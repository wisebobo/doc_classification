#!/bin/sh

PID=`ps -ef | grep "python3 train.py" | grep -v "grep" | awk '{print $2}'`

if [ ! -z "$PID" ]; then
  for p in "$PID"
  do
    kill -9 $p
  done
fi

cd /opt/python3/projects/img_class/

nohup python3 train.py >> train.log 2>&1 &
