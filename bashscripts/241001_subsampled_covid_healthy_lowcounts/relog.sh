for i in $(seq 0 3600);
do 
	for fn in $(cat all_error_logs.txt  | grep " True" | grep -v "XX_REMOVED" | awk '{print $1}'); 
	do 
		n=$(ls ../../output/*${fn}* | wc -l)
		if [[ $n -eq 46 ]]
		then 
			echo "Removing ${fn} from log"
			sed -i "s/${fn}/XX_REMOVED/g" all_error_logs.txt; 
		fi
	done
	sleep 1
done
