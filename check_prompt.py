import sqlite3
import os

db_path = "app/database/journal.db"
if not os.path.exists(db_path):
    print("Database not found")
else:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT prompt_text FROM system_prompts WHERE key='generate_big_five_analysis'")
    row = cursor.fetchone()
    if row:
        print(row['prompt_text'])
    else:
        print("Key not found")
    conn.close()
