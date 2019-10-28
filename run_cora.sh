for i in {1..10}
do

python execute_cora.py origin 1,1 8,1 0.6 0.6 140 ./out_cora/$i ./log_cora/$i

python execute_cora_sep.py train 1,4 6,1 0.6 0.6 140 ./out_cora/$i ./log_cora/$i
python execute_cora_sep.py train 1,6 4,1 0.6 0.6 140 ./out_cora/$i ./log_cora/$i
python execute_cora_sep.py train 1,4 4,4 0.6 0.6 140 ./out_cora/$i ./log_cora/$i
python execute_cora_sep.py train 1,4 6,6 0.6 0.6 140 ./out_cora/$i ./log_cora/$i
python execute_cora_sep.py train 1,8 1,1 0.6 0.6 140 ./out_cora/$i ./log_cora/$i

python execute_cora.py origin 1,1 8,1 0.6 0.6 84 ./out_cora/$i ./log_cora/$i

python execute_cora_sep.py train 1,4 6,1 0.6 0.6 84 ./out_cora/$i ./log_cora/$i
python execute_cora_sep.py train 1,6 4,1 0.6 0.6 84 ./out_cora/$i ./log_cora/$i
python execute_cora_sep.py train 1,4 4,4 0.6 0.6 84 ./out_cora/$i ./log_cora/$i
python execute_cora_sep.py train 1,4 6,6 0.6 0.6 84 ./out_cora/$i ./log_cora/$i
python execute_cora_sep.py train 1,8 1,1 0.6 0.6 84 ./out_cora/$i ./log_cora/$i

python execute_cora.py origin 1,1 8,1 0.6 0.6 28 ./out_cora/$i ./log_cora/$i

python execute_cora_sep.py train 1,4 6,1 0.6 0.6 28 ./out_cora/$i ./log_cora/$i
python execute_cora_sep.py train 1,6 4,1 0.6 0.6 28 ./out_cora/$i ./log_cora/$i
python execute_cora_sep.py train 1,4 4,4 0.6 0.6 28 ./out_cora/$i ./log_cora/$i
python execute_cora_sep.py train 1,4 6,6 0.6 0.6 28 ./out_cora/$i ./log_cora/$i
python execute_cora_sep.py train 1,8 1,1 0.6 0.6 28 ./out_cora/$i ./log_cora/$i

python execute_cora.py origin 1,1 8,1 0.6 0.6 14 ./out_cora/$i ./log_cora/$i

python execute_cora_sep.py train 1,4 6,1 0.6 0.6 14 ./out_cora/$i ./log_cora/$i
python execute_cora_sep.py train 1,6 4,1 0.6 0.6 14 ./out_cora/$i ./log_cora/$i
python execute_cora_sep.py train 1,4 4,4 0.6 0.6 14 ./out_cora/$i ./log_cora/$i
python execute_cora_sep.py train 1,4 6,6 0.6 0.6 14 ./out_cora/$i ./log_cora/$i
python execute_cora_sep.py train 1,8 1,1 0.6 0.6 14 ./out_cora/$i ./log_cora/$i

done