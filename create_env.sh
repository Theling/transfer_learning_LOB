conda create -n $1 python=3.7 anaconda -y
pip install tensorflow==2.1
conda install -n $1 keras psycopg2 -y