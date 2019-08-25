for i in {5..10}
do

python execute_cora.py origin 1,1 8,1 0.6 0.6 140 ./out/$i ./log/$i

python execute_cora.py train 3,1 6,1 0.6 0.6 140 ./out/$i ./log/$i
python execute_cora.py train 4,1 4,1 0.6 0.6 140 ./out/$i ./log/$i
python execute_cora.py train 6,1 3,1 0.6 0.6 140 ./out/$i ./log/$i
# python execute_cora.py train 6,6 3,1 0.6 0.6 140 ./out/$i ./log/$i

python execute_cora.py train_share 3,1 8,1 0.6 0.6 140 ./out/$i ./log/$i
python execute_cora.py train_share 4,1 4,1 0.6 0.6 140 ./out/$i ./log/$i
python execute_cora.py train_share 8,1 3,1 0.6 0.6 140 ./out/$i ./log/$i
# python execute_cora.py train_share 6,6 3,1 0.6 0.6 140 ./out/$i ./log/$i

python execute_cora.py random_const 3,1 8,1 0.6 0.6 140 ./out/$i ./log/$i
python execute_cora.py random_const 4,1 4,1 0.6 0.6 140 ./out/$i ./log/$i
python execute_cora.py random_const 8,1 3,1 0.6 0.6 140 ./out/$i ./log/$i
# python execute_cora.py random_const 6,6 3,1 0.6 0.6 140 ./out/$i ./log/$i


python execute_cora.py origin 1,1 8,1 0.6 0.6 70 ./out/$i ./log/$i

python execute_cora.py train 3,1 6,1 0.6 0.6 70 ./out/$i ./log/$i
python execute_cora.py train 4,1 4,1 0.6 0.6 70 ./out/$i ./log/$i
python execute_cora.py train 6,1 3,1 0.6 0.6 70 ./out/$i ./log/$i
# python execute_cora.py train 6,6 3,1 0.6 0.6 70 ./out/$i ./log/$i

python execute_cora.py train_share 3,1 8,1 0.6 0.6 70 ./out/$i ./log/$i
python execute_cora.py train_share 4,1 4,1 0.6 0.6 70 ./out/$i ./log/$i
python execute_cora.py train_share 8,1 3,1 0.6 0.6 70 ./out/$i ./log/$i
# python execute_cora.py train_share 6,6 3,1 0.6 0.6 70 ./out/$i ./log/$i

python execute_cora.py random_const 3,1 8,1 0.6 0.6 70 ./out/$i ./log/$i
python execute_cora.py random_const 4,1 4,1 0.6 0.6 70 ./out/$i ./log/$i
python execute_cora.py random_const 8,1 3,1 0.6 0.6 70 ./out/$i ./log/$i
# python execute_cora.py random_const 6,6 3,1 0.6 0.6 70 ./out/$i ./log/$i


python execute_cora.py origin 1,1 8,1 0.6 0.6 35 ./out/$i ./log/$i

python execute_cora.py train 3,1 6,1 0.6 0.6 35 ./out/$i ./log/$i
python execute_cora.py train 4,1 4,1 0.6 0.6 35 ./out/$i ./log/$i
python execute_cora.py train 6,1 3,1 0.6 0.6 35 ./out/$i ./log/$i

python execute_cora.py train_share 3,1 8,1 0.6 0.6 35 ./out/$i ./log/$i
python execute_cora.py train_share 4,1 4,1 0.6 0.6 35 ./out/$i ./log/$i
python execute_cora.py train_share 8,1 3,1 0.6 0.6 35 ./out/$i ./log/$i

python execute_cora.py random_const 3,1 8,1 0.6 0.6 35 ./out/$i ./log/$i
python execute_cora.py random_const 4,1 4,1 0.6 0.6 35 ./out/$i ./log/$i
python execute_cora.py random_const 8,1 3,1 0.6 0.6 35 ./out/$i ./log/$i


python execute_cora.py origin 1,1 8,1 0.6 0.6 14 ./out/$i ./log/$i

python execute_cora.py train 3,1 6,1 0.6 0.6 14 ./out/$i ./log/$i
python execute_cora.py train 4,1 4,1 0.6 0.6 14 ./out/$i ./log/$i
python execute_cora.py train 6,1 3,1 0.6 0.6 14 ./out/$i ./log/$i

python execute_cora.py train_share 3,1 8,1 0.6 0.6 14 ./out/$i ./log/$i
python execute_cora.py train_share 4,1 4,1 0.6 0.6 14 ./out/$i ./log/$i
python execute_cora.py train_share 8,1 3,1 0.6 0.6 14 ./out/$i ./log/$i

python execute_cora.py random_const 3,1 8,1 0.6 0.6 14 ./out/$i ./log/$i
python execute_cora.py random_const 4,1 4,1 0.6 0.6 14 ./out/$i ./log/$i
python execute_cora.py random_const 8,1 3,1 0.6 0.6 14 ./out/$i ./log/$i

done