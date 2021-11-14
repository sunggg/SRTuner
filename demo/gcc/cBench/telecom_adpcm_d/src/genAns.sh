#!/bin/bash
make -j4 CCC_OPTS="-O3"

for i in {1..20}
do
	echo ""
	echo ${i}
	./__run ${i}
	./_ccc_check_output.copy ${i}
done


for i in {1..20}
do
	echo ""
	echo ${i}
	./__run ${i}
	./_ccc_check_output.diff ${i}
	if [ -s tmp-ccc-diff ]; then
		echo "WO"
		exit 2
	fi
done

echo "ALL PASS"
