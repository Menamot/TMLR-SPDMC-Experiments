import json
import sqlite3

import numpy as np
import pandas as pd


class ExperimentDB:
    """Unified SQLite storage for SPDMC experiments.

    Manages 5 normalized tables: experiments, best_hyperparams,
    evaluation_results, class_risks, loss_functions.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()

    def close(self):
        self._conn.close()

    def _create_tables(self):
        cur = self._conn.cursor()

        cur.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tag TEXT,
            description TEXT,
            dataset TEXT NOT NULL,
            stage TEXT NOT NULL,
            n_folds INTEGER,
            n_repeats INTEGER,
            n_priors INTEGER,
            prior_sampling_method TEXT,
            prior_concentration REAL,
            feature_noise_std REAL DEFAULT 0.0,
            random_seed INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS best_hyperparams (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id INTEGER NOT NULL REFERENCES experiments(id),
            model_name TEXT NOT NULL,
            n_trials INTEGER,
            best_score REAL,
            params_json TEXT NOT NULL
        )
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS evaluation_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id INTEGER NOT NULL REFERENCES experiments(id),
            model_name TEXT NOT NULL,
            repeat_idx INTEGER NOT NULL,
            fold_idx INTEGER NOT NULL,
            data_split TEXT NOT NULL,
            prior_idx INTEGER,
            global_risk REAL NOT NULL,
            max_class_risk REAL NOT NULL,
            class_risk_range REAL NOT NULL,
            class_risk_variance REAL NOT NULL
        )
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS class_risks (
            result_id INTEGER NOT NULL REFERENCES evaluation_results(id),
            class_label INTEGER NOT NULL,
            class_risk REAL NOT NULL,
            target_prior REAL NOT NULL
        )
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS loss_functions (
            experiment_id INTEGER NOT NULL REFERENCES experiments(id),
            true_class INTEGER NOT NULL,
            predicted_class INTEGER NOT NULL,
            loss_value REAL NOT NULL
        )
        """)

        # Indexes
        cur.execute("""CREATE INDEX IF NOT EXISTS idx_eval_main ON evaluation_results (experiment_id, model_name, data_split)""")
        cur.execute("""CREATE INDEX IF NOT EXISTS idx_eval_cv ON evaluation_results (repeat_idx, fold_idx)""")
        cur.execute("""CREATE INDEX IF NOT EXISTS idx_class_risks_result ON class_risks (result_id)""")
        cur.execute("""CREATE INDEX IF NOT EXISTS idx_experiments_tag ON experiments (tag)""")
        cur.execute("""CREATE INDEX IF NOT EXISTS idx_experiments_dataset ON experiments (dataset)""")

        self._conn.commit()

    # ---- Write ----

    def create_experiment(self, dataset: str, stage: str, **kwargs) -> int:
        """Insert into experiments table, return experiment_id."""
        columns = ["dataset", "stage"]
        values = [dataset, stage]
        valid_keys = {
            "tag", "description", "n_folds", "n_repeats", "n_priors",
            "prior_sampling_method", "prior_concentration",
            "feature_noise_std", "random_seed",
        }
        for key, val in kwargs.items():
            if key in valid_keys and val is not None:
                columns.append(key)
                values.append(val)

        placeholders = ", ".join(["?"] * len(columns))
        col_str = ", ".join(columns)
        cur = self._conn.cursor()
        cur.execute(f"INSERT INTO experiments ({col_str}) VALUES ({placeholders})", values)
        self._conn.commit()
        return cur.lastrowid

    def save_hyperparams(self, experiment_id: int, model_name: str, n_trials: int, best_score: float, params: dict):
        """Insert into best_hyperparams table."""
        self._conn.execute(
            """INSERT INTO best_hyperparams (experiment_id, model_name, n_trials, best_score, params_json)
            VALUES (?, ?, ?, ?, ?)""",
            (experiment_id, model_name, n_trials, best_score, json.dumps(params, sort_keys=True)),
        )
        self._conn.commit()

    def save_results_batch(self, rows: list[dict]):
        """Batch insert into evaluation_results + class_risks."""
        cur = self._conn.cursor()
        for row in rows:
            cur.execute(
                """INSERT INTO evaluation_results
                (experiment_id, model_name, repeat_idx, fold_idx, data_split,
                 prior_idx, global_risk, max_class_risk, class_risk_range, class_risk_variance)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (row["experiment_id"], row["model_name"],
                 row["repeat_idx"], row["fold_idx"], row["data_split"],
                 row["prior_idx"], row["global_risk"], row["max_class_risk"],
                 row["class_risk_range"], row["class_risk_variance"]),
            )
            result_id = cur.lastrowid
            for cr in row.get("class_risks", []):
                cur.execute(
                    """INSERT INTO class_risks (result_id, class_label, class_risk, target_prior)
                    VALUES (?, ?, ?, ?)""",
                    (result_id, cr["class_label"], cr["class_risk"], cr["target_prior"]),
                )
        self._conn.commit()

    def save_loss_function(self, experiment_id: int, loss_matrix: np.ndarray):
        """Insert loss matrix into loss_functions table."""
        n = loss_matrix.shape[0]
        rows = []
        for i in range(n):
            for j in range(n):
                rows.append((experiment_id, i, j, float(loss_matrix[i, j])))
        self._conn.executemany(
            "INSERT INTO loss_functions (experiment_id, true_class, predicted_class, loss_value) VALUES (?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()

    # ---- Read ----

    def list_experiments(self, dataset: str = None, stage: str = None, tag: str = None) -> pd.DataFrame:
        query = "SELECT * FROM experiments WHERE 1=1"
        params = []
        if dataset is not None:
            query += " AND dataset = ?"
            params.append(dataset)
        if stage is not None:
            query += " AND stage = ?"
            params.append(stage)
        if tag is not None:
            query += " AND tag = ?"
            params.append(tag)
        query += " ORDER BY id DESC"
        return pd.read_sql_query(query, self._conn, params=params)

    def load_hyperparams(self, experiment_id: int = None, tag: str = None) -> dict:
        if experiment_id is not None:
            df = pd.read_sql_query(
                "SELECT model_name, params_json FROM best_hyperparams WHERE experiment_id = ?",
                self._conn, params=(int(experiment_id),),
            )
        elif tag is not None:
            df = pd.read_sql_query(
                """SELECT bh.model_name, bh.params_json
                FROM best_hyperparams bh JOIN experiments e ON bh.experiment_id = e.id
                WHERE e.tag = ?""",
                self._conn, params=(tag,),
            )
        else:
            raise ValueError("Must specify experiment_id or tag")
        return {row["model_name"]: json.loads(row["params_json"]) for _, row in df.iterrows()}

    def load_results(self, experiment_id: int = None, tag: str = None, dataset: str = None) -> pd.DataFrame:
        query = """
        SELECT er.*, e.tag, e.dataset, e.feature_noise_std, e.n_folds, e.n_repeats, e.n_priors
        FROM evaluation_results er JOIN experiments e ON er.experiment_id = e.id WHERE 1=1
        """
        params = []
        if experiment_id is not None:
            query += " AND er.experiment_id = ?"
            params.append(experiment_id)
        if tag is not None:
            query += " AND e.tag = ?"
            params.append(tag)
        if dataset is not None:
            query += " AND e.dataset = ?"
            params.append(dataset)
        return pd.read_sql_query(query, self._conn, params=params)

    def load_class_risks(self, experiment_id: int = None, tag: str = None) -> pd.DataFrame:
        query = """
        SELECT cr.*, er.experiment_id, er.model_name, er.repeat_idx,
               er.fold_idx, er.data_split, er.prior_idx
        FROM class_risks cr JOIN evaluation_results er ON cr.result_id = er.id WHERE 1=1
        """
        params = []
        if experiment_id is not None:
            query += " AND er.experiment_id = ?"
            params.append(experiment_id)
        if tag is not None:
            query += " AND er.experiment_id IN (SELECT id FROM experiments WHERE tag = ?)"
            params.append(tag)
        return pd.read_sql_query(query, self._conn, params=params)
