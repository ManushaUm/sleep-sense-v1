import sqlalchemy
import sys

# Connection string with URL-encoded password
DATABASE_URL = "postgresql://sleepsense_user:ceXejl%26gh4riE2%40cX%24@34.61.25.90/sleepsense"

print("Connecting to sleepsense-db...")
try:
    engine = sqlalchemy.create_engine(DATABASE_URL)
    with engine.connect() as conn:
        print("Connected successfully!")
        print("Type your SQL query and press Enter. Type 'exit' to quit.\n")
        
        while True:
            try:
                query = input("SQL> ")
                if query.strip().lower() in ("exit", "quit"):
                    break
                if not query.strip():
                    continue
                
                result = conn.execute(sqlalchemy.text(query))
                
                # If query returns rows (like SELECT)
                if result.returns_rows:
                    rows = result.fetchall()
                    keys = result.keys()
                    if not rows:
                        print("No rows returned.")
                        continue
                    
                    # Print headers
                    headers = " | ".join(keys)
                    print("\n" + headers)
                    print("-" * len(headers))
                    
                    # Print rows
                    for row in rows:
                        print(" | ".join(str(val) for val in row))
                    print(f"({len(rows)} row(s) returned)\n")
                else:
                    # For INSERT, UPDATE, DELETE, CREATE, etc.
                    conn.commit()
                    print(f"Query OK, affected {result.rowcount} row(s).\n")
                    
            except KeyboardInterrupt:
                print("\nUse 'exit' to quit.")
            except Exception as e:
                print(f"Error: {e}\n")
                
except Exception as e:
    print(f"Failed to connect to database: {e}")
