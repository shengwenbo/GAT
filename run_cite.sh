for i in {1..10}
do

python execute_citeseer.py origin 1,1 8,1 0.6 0.6 120 ./out_cite/$i ./log_cite/$i

python execute_citeseer_sep.py train 1,4 6,1 0.6 0.6 120 ./out_cite/$i ./log_cite/$i
python execute_citeseer_sep.py train 1,6 4,1 0.6 0.6 120 ./out_cite/$i ./log_cite/$i
python execute_citeseer_sep.py train 1,4 4,4 0.6 0.6 120 ./out_cite/$i ./log_cite/$i
python execute_citeseer_sep.py train 1,4 6,6 0.6 0.6 120 ./out_cite/$i ./log_cite/$i
python execute_citeseer_sep.py train 1,8 1,1 0.6 0.6 120 ./out_cite/$i ./log_cite/$i

python execute_citeseer.py origin 1,1 8,1 0.6 0.6 36 ./out_cite/$i ./log_cite/$i

python execute_citeseer_sep.py train 1,4 6,1 0.6 0.6 36 ./out_cite/$i ./log_cite/$i
python execute_citeseer_sep.py train 1,6 4,1 0.6 0.6 36 ./out_cite/$i ./log_cite/$i
python execute_citeseer_sep.py train 1,4 4,4 0.6 0.6 36 ./out_cite/$i ./log_cite/$i
python execute_citeseer_sep.py train 1,4 6,6 0.6 0.6 36 ./out_cite/$i ./log_cite/$i
python execute_citeseer_sep.py train 1,8 1,1 0.6 0.6 36 ./out_cite/$i ./log_cite/$i

python execute_citeseer.py origin 1,1 8,1 0.6 0.6 18 ./out_cite/$i ./log_cite/$i

python execute_citeseer_sep.py train 1,4 6,1 0.6 0.6 18 ./out_cite/$i ./log_cite/$i
python execute_citeseer_sep.py train 1,6 4,1 0.6 0.6 18 ./out_cite/$i ./log_cite/$i
python execute_citeseer_sep.py train 1,4 4,4 0.6 0.6 18 ./out_cite/$i ./log_cite/$i
python execute_citeseer_sep.py train 1,4 6,6 0.6 0.6 18 ./out_cite/$i ./log_cite/$i
python execute_citeseer_sep.py train 1,8 1,1 0.6 0.6 18 ./out_cite/$i ./log_cite/$i


python execute_citeseer.py origin 1,1 8,1 0.6 0.6 12 ./out_cite/$i ./log_cite/$i

python execute_citeseer_sep.py train 1,4 6,1 0.6 0.6 12 ./out_cite/$i ./log_cite/$i
python execute_citeseer_sep.py train 1,6 4,1 0.6 0.6 12 ./out_cite/$i ./log_cite/$i
python execute_citeseer_sep.py train 1,4 4,4 0.6 0.6 12 ./out_cite/$i ./log_cite/$i
python execute_citeseer_sep.py train 1,4 6,6 0.6 0.6 12 ./out_cite/$i ./log_cite/$i
python execute_citeseer_sep.py train 1,8 1,1 0.6 0.6 12 ./out_cite/$i ./log_cite/$i


done