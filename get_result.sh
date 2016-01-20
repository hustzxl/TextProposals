#########################################################################
# File Name: get_result.sh
# Author: ma6174
# mail: ma6174@163.com
# Created Time: Wed 20 Jan 2016 04:44:52 PM CST
#########################################################################
#!/bin/bash
for i in `cat ../Data/svt/test.xml | grep imageName | cut -d '>' -f 2 | cut -d '<' -f 1 | cut -d '/' -f 2 | cut -d '.' -f 1 `; do echo $i; ./img2hierarchy ../Data//svt/img/$i.jpg 13 > ../result/$i; done
