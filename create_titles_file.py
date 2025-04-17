import psycopg2
from datetime import datetime, timedelta

# Calculate timestamp for 5 years ago
ten_years_ago = datetime.now() - timedelta(days=2*365)
ten_years_ago_str = ten_years_ago.strftime('%Y-%m-%d %H:%M:%S')

# Connect to your PostgreSQL database
conn = psycopg2.connect(
    host="178.156.142.230",
    port="5432",
    database="hd64m1ki",
    user="sy91dhb",
    password="g5t49ao"
)

cur = conn.cursor()

# Query to fetch titles from stories posted in the last 5 years
cur.execute("""
    SELECT text 
    FROM hacker_news.items 
    WHERE type = 'comment' 
    AND text IS NOT NULL
    AND time >= %s;
""", (ten_years_ago_str,))

rows = cur.fetchall()

# Join titles into a single corpus
hn_titles = ' '.join(row[0] for row in rows)

# Close the connection
cur.close()
conn.close()

# Save to file
with open('hacker_news_titles_10yrs.txt', 'w', encoding='utf-8') as f:
    f.write(hn_titles)

# Preview first 500 chars
print(hn_titles[:500])

