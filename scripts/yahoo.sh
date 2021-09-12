python -u train.py yahoo anomaly_0 --loader anomaly --repr-dims 320 --max-threads 8 --seed 1 --eval
python -u train.py yahoo anomaly_1 --loader anomaly --repr-dims 320 --max-threads 8 --seed 2 --eval
python -u train.py yahoo anomaly_2 --loader anomaly --repr-dims 320 --max-threads 8 --seed 3 --eval

python -u train.py yahoo anomaly_coldstart_0 --loader anomaly_coldstart --repr-dims 320 --max-threads 8 --seed 1 --eval
python -u train.py yahoo anomaly_coldstart_1 --loader anomaly_coldstart --repr-dims 320 --max-threads 8 --seed 2 --eval
python -u train.py yahoo anomaly_coldstart_2 --loader anomaly_coldstart --repr-dims 320 --max-threads 8 --seed 3 --eval
