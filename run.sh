
ARRAY=()
for i in `seq 1 4`;
do
    python mcws.py &> log_$i.out &
    echo "Submitted job "$i
    ARRAY+=($!)
    if [ ${#ARRAY[@]} -gt 1 ];
    then
	echo "Running "${#ARRAY[@]}" jobs"
	PID_TO_KILL=${ARRAY[0]}
	echo -e "\tWaiting for job "$PID_TO_KILL
	wait $PID_TO_KILL
	unset ARRAY[0]
	ARRAY=( "${ARRAY[@]}" )
	sleep 1
    fi
done
