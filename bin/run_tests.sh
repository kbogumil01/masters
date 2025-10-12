#! /bin/bash
#


run_test()
{
	local model=$1
	local dataset=$2
	local logfolder=$3
	for folder in $dataset/*; do
		logfile=$logfolder/$(echo "$folder" | sed 's/\//_/g')
		python -m enhancer -t -c $model -x $folder > $logfile
	done
}


run_test ${1} ${2} ${3}
