# multivar
python -u train.py electricity forecast_multivar --archive forecast_csv --repr-dims 320 --max-threads 8 --seed 42 --eval

# univar
python -u train.py electricity forecast_univar --archive forecast_csv_univar --repr-dims 320 --max-threads 8 --seed 42 --eval
