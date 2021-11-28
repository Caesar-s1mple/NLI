# NLI

start training as

```
python main.py --policy Deberta_NLI
	       --epochs 50 (modifiable)
	       --batch-size 4 (modifiable)
	       --resume (optional)
```

if you just want to test, place all files in folder 'deberta-large-mnli' into 'res/Deberta_NLI/best', then do as following

```
python main.py --policy Deberta_NLI
	       --only-test
```

