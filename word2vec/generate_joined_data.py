from datasets import load_dataset

text8_ds = load_dataset("afmck/text8")
text8_text = text8_ds['train'][0]['text']  # It's just one huge string

with open("hacker_news_comments_2yrs.txt", "r", encoding="utf-8") as f:
    hackernews_text = f.read()

combined_text = text8_text.strip() + " " + hackernews_text.strip()

with open("text8_plus_hn2.txt", "w", encoding="utf-8") as f:
    f.write(combined_text)
