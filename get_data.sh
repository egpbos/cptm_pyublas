#!/bin/bash
# Download Dilipad data and extract the FoLiA files.
# Usage: ./get_data.sh <dir out> 
# 2015-07-01 j.vanderzwaan@esciencecenter.nl

# Create output directory if it doesn't exist  
[[ -d "$1" ]] || mkdir "$1"
[[ -d "$1/download/" ]] || mkdir "$1/download/"

cd "$1/download/"

# Download data
wget http://ode.politicalmashup.nl/data/summarise/folia/proc-ob-19992000.tar.gz
wget http://ode.politicalmashup.nl/data/summarise/folia/proc-ob-20002001.tar.gz
wget http://ode.politicalmashup.nl/data/summarise/folia/proc-ob-20012002.tar.gz
wget http://ode.politicalmashup.nl/data/summarise/folia/proc-ob-20022003.tar.gz
wget http://ode.politicalmashup.nl/data/summarise/folia/proc-ob-20032004.tar.gz
wget http://ode.politicalmashup.nl/data/summarise/folia/proc-ob-20042005.tar.gz
wget http://ode.politicalmashup.nl/data/summarise/folia/proc-ob-20052006.tar.gz
wget http://ode.politicalmashup.nl/data/summarise/folia/proc-ob-20062007.tar.gz
wget http://ode.politicalmashup.nl/data/summarise/folia/proc-ob-20072008.tar.gz
wget http://ode.politicalmashup.nl/data/summarise/folia/proc-ob-20082009.tar.gz
wget http://ode.politicalmashup.nl/data/summarise/folia/proc-ob-20092010.tar.gz
wget http://ode.politicalmashup.nl/data/summarise/folia/proc-ob-20102011.tar.gz
wget http://ode.politicalmashup.nl/data/summarise/folia/proc-ob-20112012.tar.gz

cd ..

# Unpack data
for f in download/*.tar.gz
do
  tar -zxvf "$f" --wildcards '*/data_folia'
done
