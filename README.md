pip install -r requirements.txt

mkdir pd_data 
mkdir results 
mkdir pp_logging 
mkdir data 
cp _datasets.json datasets.json 
python utils/db.py 
