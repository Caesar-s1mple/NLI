import csv

csv_file = open('./res/res.csv', 'w', encoding='utf-8', newline='')
fieldnames = ['premise', 'hypothesis', 'score', 'relation', 'h', 't']
csv_writter = csv.DictWriter(csv_file, fieldnames=fieldnames)
csv_writter.writeheader()

with open('./res/res.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        csv_writter.writerow(eval(line))
