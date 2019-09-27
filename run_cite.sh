for i in {1..10}
do

python execute_citeseer.py origin 1,1 8,1 0.6 0.6 120 ./out_cite/$i ./log_cite/$i

python execute_citeseer_sep.py train 4,1 6,1 0.6 0.6 120 ./out_cite/$i ./log_cite/$i
python execute_citeseer_sep.py train 6,1 4,1 0.6 0.6 120 ./out_cite/$i ./log_cite/$i
python execute_citeseer_sep.py train 4,4 6,1 0.6 0.6 120 ./out_cite/$i ./log_cite/$i
python execute_citeseer_sep.py train 6,6 4,1 0.6 0.6 120 ./out_cite/$i ./log_cite/$i
python execute_citeseer_sep.py train 6,6 6,1 0.6 0.6 120 ./out_cite/$i ./log_cite/$i

python execute_citeseer.py origin 1,1 8,1 0.6 0.6 60 ./out_cite/$i ./log_cite/$i

python execute_citeseer_sep.py train 4,1 6,1 0.6 0.6 60 ./out_cite/$i ./log_cite/$i
python execute_citeseer_sep.py train 6,1 4,1 0.6 0.6 60 ./out_cite/$i ./log_cite/$i
python execute_citeseer_sep.py train 4,4 6,1 0.6 0.6 60 ./out_cite/$i ./log_cite/$i
python execute_citeseer_sep.py train 6,6 4,1 0.6 0.6 60 ./out_cite/$i ./log_cite/$i
python execute_citeseer_sep.py train 6,6 6,1 0.6 0.6 60 ./out_cite/$i ./log_cite/$i

python execute_citeseer.py origin 1,1 8,1 0.6 0.6 30 ./out_cite/$i ./log_cite/$i

python execute_citeseer_sep.py train 4,1 6,1 0.6 0.6 30 ./out_cite/$i ./log_cite/$i
python execute_citeseer_sep.py train 6,1 4,1 0.6 0.6 30 ./out_cite/$i ./log_cite/$i
python execute_citeseer_sep.py train 4,4 6,1 0.6 0.6 30 ./out_cite/$i ./log_cite/$i
python execute_citeseer_sep.py train 6,6 4,1 0.6 0.6 30 ./out_cite/$i ./log_cite/$i
python execute_citeseer_sep.py train 6,6 6,1 0.6 0.6 30 ./out_cite/$i ./log_cite/$i


python execute_citeseer.py origin 1,1 8,1 0.6 0.6 18 ./out_cite/$i ./log_cite/$i

python execute_citeseer_sep.py train 4,1 6,1 0.6 0.6 18 ./out_cite/$i ./log_cite/$i
python execute_citeseer_sep.py train 6,1 4,1 0.6 0.6 18 ./out_cite/$i ./log_cite/$i
python execute_citeseer_sep.py train 4,4 6,1 0.6 0.6 18 ./out_cite/$i ./log_cite/$i
python execute_citeseer_sep.py train 6,6 4,1 0.6 0.6 18 ./out_cite/$i ./log_cite/$i
python execute_citeseer_sep.py train 6,6 6,1 0.6 0.6 18 ./out_cite/$i ./log_cite/$i

python execute_citeseer.py origin 1,1 8,1 0.6 0.6 10 ./out_cite/$i ./log_cite/$i

python execute_citeseer_sep.py train 4,1 6,1 0.6 0.6 10 ./out_cite/$i ./log_cite/$i
python execute_citeseer_sep.py train 6,1 4,1 0.6 0.6 10 ./out_cite/$i ./log_cite/$i
python execute_citeseer_sep.py train 4,4 6,1 0.6 0.6 10 ./out_cite/$i ./log_cite/$i
python execute_citeseer_sep.py train 6,6 4,1 0.6 0.6 10 ./out_cite/$i ./log_cite/$i
python execute_citeseer_sep.py train 6,6 6,1 0.6 0.6 10 ./out_cite/$i ./log_cite/$i


done