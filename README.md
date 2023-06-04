# CycleGAN
CycleGAN from scratch (pytorch)

> [Paper Review](https://inhopp.github.io/paper/Paper18/)

| Input | Gogh | Monet |
|:-:|:-:|:-:|
| ![temp](https://github.com/inhopp/inhopp/assets/96368476/1970e94e-2e5e-437d-9c8c-60abdbba31c3) | ![output](https://github.com/inhopp/inhopp/assets/96368476/4c532445-7241-4fba-93ff-5f3ed473a2c7) | ![output2](https://github.com/inhopp/inhopp/assets/96368476/48cdbcde-0ed8-4b9d-9af0-b1433cb15afc) |
## Repository Directory 

``` python 
├── CycleGAN
        ├── datasets
        │    
        ├── data
        │     ├── __init__.py
        │     └── dataset.py
        ├── option.py
        ├── model.py
        ├── train.py
        ├── inference.py
        └── README.md
```

- `data/__init__.py` : dataset loader
- `data/dataset.py` : data preprocess & get item
- `model.py` : Define block and construct Model
- `option.py` : Environment setting

<br>


## Tutoral

### Clone repo and install depenency

``` python
# Clone this repo and install dependency
git clone https://github.com/inhopp/CycleGAN.git
```

<br>


### train
``` python
python3 train.py
    --device {}(defautl: cpu) \
    --lr {}(default: 0.0002) \
    --n_epoch {}(default: 200) \
    --num_workers {}(default: 4) \
    --batch_size {}(default: 4) \ 
    --eval_batch_size {}(default: 4)
```

### testset inference
``` python
python3 inference.py
    --device {}(defautl: cpu) \
    --num_workers {}(default: 4) \
    --eval_batch_size {}(default: 4)
```


<br>


#### Main Reference
https://github.com/aladdinpersson/Machine-Learning-Collection