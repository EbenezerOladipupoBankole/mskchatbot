import json

# Read the JSON file
with open('hymn_book.json', 'r', encoding='utf-8') as f:
    hymns = json.load(f)

# Update numbers sequentially
for i, hymn in enumerate(hymns, start=1):
    hymn['number'] = str(i)

# Write back to the file
with open('hymn_book.json', 'w', encoding='utf-8') as f:
    json.dump(hymns, f, ensure_ascii=False, indent=2)