# multivar
python -u train.py electricity forecast_multivar --loader forecast_csv --repr-dims 320 --max-threads 8 --seed 42 --eval

# univar
python -u train.py electricity forecast_univar --loader forecast_csv_univar --repr-dims 320 --max-threads 8 --seed 42 --eval
