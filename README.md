# NLI

- download dataset from 

  https://github.com/thunlp/OpenNRE/tree/master/benchmark

- download 'pytorch_model.bin' from

  https://huggingface.co/microsoft/deberta-large-mnli/tree/main

- start training as

  ```
  python main.py --policy Deberta_NLI
  	       --epochs 50 (modifiable)
  	       --batch-size 4 (modifiable)
  	       --resume (optional)
  ```

  

- if you just want to test, place all files in folder 'deberta-large-mnli' into 'res/Deberta_NLI/best', then do as following

  ```
  python main.py --policy Deberta_NLI
  	       --only-test
  ```

  

