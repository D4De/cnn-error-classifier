import sqlite3
import os
from typing import List

from analyzed_tensor import AnalyzedTensor


def execute_db_script(sql_script : str, db_path : str):
    # Connect to the SQLite database (creates a new database if it doesn't exist)


    with sqlite3.connect(db_path) as conn:
       # Create a cursor object to execute SQL statements
        cursor = conn.cursor()

        # Execute the SQL code to create the tables
        cursor.executescript(sql_script)


def create_db(db_path : str):
    file_directory, file_name = os.path.split(os.path.realpath(__file__))
    sql_path = os.path.join(file_directory, 'sql', 'create_database.sql')
    
    with open(sql_path, 'r') as f:
        sql_script = f.read()
    execute_db_script(sql_script, db_path)

def put_experiment_data(db_path : str, results : List[AnalyzedTensor]):
    experiments = [exp.as_insert_param_list() for exp in results]
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        file_directory, file_name = os.path.split(os.path.realpath(__file__))

        insert_script_path = os.path.join(file_directory, 'sql', 'insert_experiment.sql')
        with open(insert_script_path, 'r') as f:
            insert_prep_statement = f.read()

        cursor.executemany(insert_prep_statement, experiments)


def delete_db(db_path : str):
    if os.path.exists(db_path):
        os.remove(db_path)

