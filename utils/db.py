import sqlite3
from sqlite3 import Error

import logging


db_location = "pd_data/central.db"

dist_dataframe_fn = "pd_data/dist_df.pkl"
cls_dataframe_fn = "pd_data/cls_df.pkl"
model_dataframe_fn = "pd_data/model_df.pkl"
perf_dataframe_fn = "pd_data/performance.pkl"


def create_connection(db_file):
    """
    Create database connection for SQLite Database
    specified by db_file
    """

    conn = None

    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)

    return conn


def create_table(conn, create_table_sql):
    """
    create a table from the create_table_sql statement
    :param conn: Connection Object
    :param: create_table_sql: a CREATE TABLE statement
    """

    try: 
        c = conn.cursor()
        c.execute(create_table_sql)
    except Error as e:
        print(e)


def insert_into_table(table_name, data_dict, conn=None):
    """
    Insert into tabel creates sql command based on
    dict
    """

    if "id" in data_dict and check_existence(table_name, {"id": data_dict["id"]}, conn=conn):
        print("Found Match %s: id: %s" %(table_name, data_dict["id"]))

        logging.info("Found Match %s: id: %s" %(table_name, data_dict["id"]))
        logging.info("INSERT ANYWAY")
        if conn is None:
            conn = create_connection(db_location)

        sql = "REPLACE INTO %s(" % table_name
        keys = ", ".join(data_dict.keys())
        sql += keys + ")"

        sql += " VALUES("
        sql += ", ".join(["?" for _ in data_dict.keys()]) + ")"

        value_tup = tuple(data_dict.values())


        cur = conn.cursor()
        cur.execute(sql, value_tup)
        conn.commit()
        return False

    else:
        if conn is None:
            conn = create_connection(db_location)

        sql = "INSERT INTO %s(" % table_name
        keys = ", ".join(data_dict.keys())
        sql += keys + ")"

        sql += " VALUES("
        sql += ", ".join(["?" for _ in data_dict.keys()]) + ")"

        value_tup = tuple(data_dict.values())


        cur = conn.cursor()
        cur.execute(sql, value_tup)
        conn.commit()
        return True


def get_selection_table(table_name, fields=None, match_e=None, conn=None):
    """
    Get selection from db
    """

    if conn is None:
        conn = create_connection(db_location)

    conn.row_factory  = sqlite3.Row
    sql = "SELECT "

    if fields is not None:

        sql_fields = ", ".join(fields)
        sql += sql_fields + " FROM " + table_name + " "
    else:
        sql += "* FROM " + table_name 
    
    if match_e is not None:
        sql_matches = " ".join(["%s = ?" %(k) for k in match_e.keys()])
        match_tup = tuple(match_e.values())
        sql += " WHERE " + sql_matches
        
        cur = conn.cursor()
        rows = cur.execute(sql, match_tup).fetchall()

    else:
        
        cur = conn.cursor()
        rows = cur.execute(sql).fetchall()


    return [ dict(row) for row in rows]

def check_existence(table_name, match_e, conn=None):
    """
    check existence of identifying elements
    """
    results = get_selection_table(table_name, match_e=match_e, conn=conn)

    return len(results) > 0


sql_create_results = """ CREATE TABLE IF NOT EXISTS results (
                             id integer PRIMARY KEY,
                             source text NOT NULL,
                             target text NOT NULL,
                             network text NOT NULL,
                             layer_probe text,
                             disc_network text,
                             class_training text,
                             disc_training text,
                             disc_acc real,
                             a_proxy real,
                             model_loc, text,
                             split real,
                             downsize real,
                             date datetime);
                    """


create_dist_results = """ CREATE TABLE IF NOT EXISTS distances (
                              id text PRIMARY KEY,
                              base_ds text,
                              dist_type text,
                              target_ds text,
                              model text,
                              accuracy real,
                              auc real,
                              brier real, 
                              ece real, 
                              test_avg real,
                              loss real,
                              score real,
                              mse real,
                              mae real);
                      """



create_bootstrap_results = """ CREATE TABLE IF NOT EXISTS bootstrap (
                               id text PRIMARY KEY,
                               filename text,
                               dataset text);
                           """

create_cls_results = """ CREATE TABLE IF NOT EXISTS classifications (
                             id text PRIMARY KEY,
                             base_ds text,
                             target_ds text,
                             model text,
                             dist_type text,
                             accuracy real,
                             auc real,
                             brier real,
                             ece real,
                             test_avg real,
                             loss real,
                             entropy real);
                     """

create_perf_results = """ CREATE TABLE IF NOT EXISTS performances (
                             id integer PRIMARY KEY,
                             base_ds text,
                             target_ds text,
                             model text,
                             token text,
                             base_token text,
                             cal_token text,
                             cal_alg text,
                             base_acc real,
                             gap real,
                             pred_gap real,
                             error real,
                             accuracy real,
                             pred_acc real,
                             auc real);
                      """

add_cols_cls_results_1 = """
                          ALTER TABLE classifications 
                          ADD COLUMN target_ds text;
                       """

add_cols_cls_results_2 = """
                          ALTER TABLE classifications 
                          ADD COLUMN model text;
                       """

add_cols_cls_results_3 = """
                          ALTER TABLE classifications 
                          ADD COLUMN base_ds text;
                       """


add_cols_dist_results_1 = """
                          ALTER TABLE distances 
                          ADD COLUMN target_ds text;
                       """

add_cols_dist_results_2 = """
                          ALTER TABLE distances 
                          ADD COLUMN model text;
                       """

add_cols_dist_results_3 = """
                          ALTER TABLE distances 
                          ADD COLUMN base_ds text;
                       """

add_cols_dist_results_4 = """
                          ALTER TABLE distances 
                          ADD COLUMN dist_type text;
                       """

conn = create_connection(db_location)

create_table(conn, create_dist_results)
create_table(conn, create_cls_results)
create_table(conn, create_perf_results)


