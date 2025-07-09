import sqlite3

class DataFrameAlternative:
    """Pandas DataFrame'e basit alternatif"""
    def __init__(self, data=None, columns=None):
        if data is None:
            self.data = []
            self.columns = columns or []
        else:
            self.data = data
            self.columns = columns or list(range(len(data[0]) if data else []))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, key):
        if isinstance(key, str) and key in self.columns:
            col_idx = self.columns.index(key)
            return ColumnData([row[col_idx] for row in self.data])
        return self.data[key]
    
    @property
    def empty(self):
        return len(self.data) == 0
    
    def tail(self, n):
        new_data = self.data[-n:] if n <= len(self.data) else self.data
        return DataFrameAlternative(new_data, self.columns)

class ColumnData:
    """Pandas Series'e basit alternatif"""
    def __init__(self, values):
        self.values = values
    
    def __getitem__(self, index):
        return self.values[index]
    
    def __len__(self):
        return len(self.values)
    
    def mean(self):
        import statistics
        return statistics.mean(self.values) if self.values else 0

def load_data_from_sqlite(db_path="jetx_data.db", limit=None):
    """
    SQLite veritabanından JetX verilerini yükler
    
    Args:
        db_path: SQLite veritabanı dosya yolu (.db uzantılı)
        limit: Yüklenecek son kayıt sayısı (None=tümü)
    
    Returns:
        DataFrameAlternative: Yüklenen veriler
    """
    conn = sqlite3.connect(db_path)
    
    # Tablo yoksa oluştur
    conn.execute('''
    CREATE TABLE IF NOT EXISTS jetx_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        value REAL NOT NULL
    )
    ''')
    
    # Predictions tablosu da oluştur
    conn.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        predicted_value REAL,
        confidence_score REAL,
        above_threshold INTEGER,
        actual_value REAL,
        was_correct INTEGER
    )
    ''')
    
    # Verileri çek
    if limit:
        query = f"SELECT * FROM jetx_results ORDER BY id DESC LIMIT {limit}"
    else:
        query = "SELECT * FROM jetx_results ORDER BY id"
    
    cursor = conn.cursor()
    cursor.execute(query)
    rows = cursor.fetchall()
    
    # Column names
    column_names = [description[0] for description in cursor.description]
    
    conn.close()
    
    # DataFrameAlternative oluştur
    df = DataFrameAlternative(rows, column_names)
    return df

def save_result_to_sqlite(value, db_path="jetx_data.db"):
    """
    Yeni bir JetX sonucunu veritabanına kaydeder
    
    Args:
        value: JetX oyun sonucu (katsayı)
        db_path: SQLite veritabanı dosya yolu
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Tablo yoksa oluştur
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS jetx_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        value REAL NOT NULL
    )
    ''')
    
    cursor.execute('''
    INSERT INTO jetx_results (value) VALUES (?)
    ''', (value,))
    
    conn.commit()
    record_id = cursor.lastrowid
    conn.close()
    
    return record_id

def save_prediction_to_sqlite(prediction_data, db_path="jetx_data.db"):
    """
    Tahmin sonuçlarını SQLite veritabanına kaydeder
    
    Args:
        prediction_data: Kaydedilecek tahmin verisi (dict)
        db_path: SQLite veritabanı dosya yolu
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Tablo yoksa oluştur
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        predicted_value REAL,
        confidence_score REAL,
        above_threshold INTEGER,
        actual_value REAL,
        was_correct INTEGER
    )
    ''')
    
    cursor.execute('''
    INSERT INTO predictions 
    (predicted_value, confidence_score, above_threshold)
    VALUES (?, ?, ?)
    ''', (
        prediction_data['predicted_value'],
        prediction_data['confidence_score'],
        1 if prediction_data['above_threshold'] else 0
    ))
    
    conn.commit()
    record_id = cursor.lastrowid
    conn.close()
    
    return record_id

def update_prediction_result(prediction_id, actual_value, db_path="jetx_data.db"):
    """
    Tahmin sonucunu günceller (gerçek değer öğrenildiğinde)
    
    Args:
        prediction_id: Güncellenecek tahmin ID'si
        actual_value: Gerçekleşen JetX değeri
        db_path: SQLite veritabanı dosya yolu
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Önce mevcut tahmini çek
    cursor.execute("SELECT above_threshold FROM predictions WHERE id=?", (prediction_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return False
    
    above_threshold = row[0]
    was_correct = 1 if (above_threshold == 1 and actual_value >= 1.5) or \
                       (above_threshold == 0 and actual_value < 1.5) else 0
    
    # Tahmini güncelle
    cursor.execute('''
    UPDATE predictions 
    SET actual_value=?, was_correct=? 
    WHERE id=?
    ''', (actual_value, was_correct, prediction_id))
    
    conn.commit()
    conn.close()
    
    return True
