for i in {1..10}
do

python execute_pubmed_sparse.py 60 ./out_pub/$i ./log_pub/$i

# python execute_pubmed_sep.py train 4,1 6,6 0.6 0.6 60 ./out_pub/$i ./log_pub/$i
# python execute_pubmed_sep.py train 6,1 4,4 0.6 0.6 60 ./out_pub/$i ./log_pub/$i
# python execute_pubmed_sep.py train 4,4 4,4 0.6 0.6 60 ./out_pub/$i ./log_pub/$i
# python execute_pubmed_sep.py train 4,4 1,1 0.6 0.6 60 ./out_pub/$i ./log_pub/$i
# python execute_pubmed_sep.py train 8,8 1,1 0.6 0.6 60 ./out_pub/$i ./log_pub/$i

python execute_pubmed_sparse.py 20 ./out_pub/$i ./log_pub/$i

# python execute_pubmed_sep.py train 4,1 6,6 0.6 0.6 20 ./out_pub/$i ./log_pub/$i
# python execute_pubmed_sep.py train 6,1 4,4 0.6 0.6 20 ./out_pub/$i ./log_pub/$i
# python execute_pubmed_sep.py train 4,4 4,4 0.6 0.6 20 ./out_pub/$i ./log_pub/$i
# python execute_pubmed_sep.py train 4,4 1,1 0.6 0.6 20 ./out_pub/$i ./log_pub/$i
# python execute_pubmed_sep.py train 8,8 1,1 0.6 0.6 20 ./out_pub/$i ./log_pub/$i

python execute_pubmed_sparse.py 10 ./out_pub/$i ./log_pub/$i

# python execute_pubmed_sep.py train 4,1 6,6 0.6 0.6 10 ./out_pub/$i ./log_pub/$i
# python execute_pubmed_sep.py train 6,1 4,4 0.6 0.6 10 ./out_pub/$i ./log_pub/$i
# python execute_pubmed_sep.py train 4,4 4,4 0.6 0.6 10 ./out_pub/$i ./log_pub/$i
# python execute_pubmed_sep.py train 4,4 1,1 0.6 0.6 10 ./out_pub/$i ./log_pub/$i
# python execute_pubmed_sep.py train 8,8 1,1 0.6 0.6 10 ./out_pub/$i ./log_pub/$i

python execute_pubmed_sparse.py 6 ./out_pub/$i ./log_pub/$i

# python execute_pubmed_sep.py train 4,1 6,6 0.6 0.6 6 ./out_pub/$i ./log_pub/$i
# python execute_pubmed_sep.py train 6,1 4,4 0.6 0.6 6 ./out_pub/$i ./log_pub/$i
# python execute_pubmed_sep.py train 4,4 4,4 0.6 0.6 6 ./out_pub/$i ./log_pub/$i
# python execute_pubmed_sep.py train 4,4 1,1 0.6 0.6 6 ./out_pub/$i ./log_pub/$i
# python execute_pubmed_sep.py train 8,8 1,1 0.6 0.6 6 ./out_pub/$i ./log_pub/$i

done