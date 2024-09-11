touch $1/all_error_logs.txt
for f in $(ls $1/*.sh.e*);  do
  name="${f%.sh.*}";
  job_id="${f##*.sh.e}";
  file_content=$(cat "${f}");
  if echo "$file_content" | grep -qi "error";
  then
    error_bool="True";
    tail_file=$(tail -n 5 "${f}");
    echo "${name} ${job_id} ${error_bool}" >> $1/all_error_logs.txt;
    echo "${tail_file}" >> $1/all_error_logs.txt;
  else
    error_bool="False";
    echo "No errors for ${name} ${job_id}. Removing log"
    rm $1/*${job_id}*
    echo "${name} ${job_id} ${error_bool}" >> $1/all_error_logs.txt;
  fi;
  echo " " >> $1/all_error_logs.txt;
done