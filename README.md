# NLI

- download dataset from 

  https://github.com/thunlp/OpenNRE/tree/master/benchmark

  then place them into folder 'dataset'

- download 'pytorch_model.bin' from

  https://huggingface.co/microsoft/deberta-large-mnli/tree/main

  then place it into folder 'deberta-large-mnli'

- start training as

  ```
  python main.py --policy Deberta_NLI
  	       --epochs 50 (modifiable)
  	       --batch-size 4 (modifiable)
  	       --resume (optional)
  ```

  

- if you just want to do inferring, you can do as following

  ```
  python main.py --policy Deberta_NLI
  	       --only-infer
  ```

