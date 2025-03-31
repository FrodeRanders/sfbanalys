# sfbanalys
Analys av Socialförsäkringsbalken med olika AI-verktyg.

Här försöker jag bland annat berika en kunskapsgraf, menad att fånga lagstiftningen. 

Én frågeställning handlar om att förstå vilka olika kategorier av text som finns; 
* läsanvisningar, 
* definition och beskrivning av begrepp, 
* regler (kring rätten till, kring beräkning och kring samordning), 
* osv.

## Setup

Såhär riggar du Neo4j för körning:
```
➜  docker pull neo4j            
➜  docker run -d --name neo4j -p7474:7474 -p7687:7687 -e NEO4J_AUTH=neo4j/sfbsfbsfb -v ./data:/data neo4j
```

Python 3.12 och NumPy 1.26 lirar dåligt, så vi måste använda Python 3.11.
```
➜  python3.11 -m venv env
➜  source env/bin/activate
➜  python -m pip install -r requirements.txt
```

## Köra

```
➜  python analys/load_graph.py
➜  python analys/graph_extract.py
➜  python analys/classify.py
Data(x=[2959, 768], edge_index=[2, 69348])
Epoch 001 | Loss: 1.0980 | Train Acc: 0.4819 | Val Acc: 0.4286
Epoch 010 | Loss: 0.8702 | Train Acc: 0.6386 | Val Acc: 0.5714
Epoch 020 | Loss: 0.7710 | Train Acc: 0.6265 | Val Acc: 0.6190
Epoch 030 | Loss: 0.7354 | Train Acc: 0.6867 | Val Acc: 0.6190
Epoch 040 | Loss: 0.7014 | Train Acc: 0.6747 | Val Acc: 0.5714
Epoch 050 | Loss: 0.6726 | Train Acc: 0.6747 | Val Acc: 0.5714
Epoch 060 | Loss: 0.6487 | Train Acc: 0.6627 | Val Acc: 0.5238
Epoch 070 | Loss: 0.6301 | Train Acc: 0.6867 | Val Acc: 0.6667
Epoch 080 | Loss: 0.6100 | Train Acc: 0.7229 | Val Acc: 0.6190
Epoch 090 | Loss: 0.6099 | Train Acc: 0.7470 | Val Acc: 0.6190
Epoch 100 | Loss: 0.5926 | Train Acc: 0.7711 | Val Acc: 0.6667
Epoch 110 | Loss: 0.5838 | Train Acc: 0.7229 | Val Acc: 0.6667
Epoch 120 | Loss: 0.5789 | Train Acc: 0.7470 | Val Acc: 0.6667
Epoch 130 | Loss: 0.5959 | Train Acc: 0.7470 | Val Acc: 0.6190
Epoch 140 | Loss: 0.5722 | Train Acc: 0.7470 | Val Acc: 0.6667
Epoch 150 | Loss: 0.5588 | Train Acc: 0.6627 | Val Acc: 0.6667
Epoch 160 | Loss: 0.5687 | Train Acc: 0.7470 | Val Acc: 0.6667
Epoch 170 | Loss: 0.5588 | Train Acc: 0.7470 | Val Acc: 0.7143
Epoch 180 | Loss: 0.5464 | Train Acc: 0.7470 | Val Acc: 0.6190
Epoch 190 | Loss: 0.5610 | Train Acc: 0.6867 | Val Acc: 0.5714
Epoch 200 | Loss: 0.5783 | Train Acc: 0.7590 | Val Acc: 0.7143
--------------------------------------------------------------------------------
[[4 0 1]
 [2 2 3]
 [0 0 9]]
              precision    recall  f1-score   support

 beskrivning       0.67      0.80      0.73         5
  definition       1.00      0.29      0.44         7
       regel       0.69      1.00      0.82         9

    accuracy                           0.71        21
   macro avg       0.79      0.70      0.66        21
weighted avg       0.79      0.71      0.67        21

```
Vårt Graph Neural Network (GNN) lär sig! Att 'Loss' minskar från ~1.09 
till ~0.58 betyder att modellen konvergerar!

Vi ser en god utveckling av 'Train Acc' (tränings-exakthet), som når ~75% 
och mäter modellens förmåga att fånga preparerad (labeled) data.

Att 'Val Acc' (validerings-exakthet) når 62-67% (toppar kring ~71%) betyder
att modellen lyckas generalisera ganska bra för inte tidigare sett data.

Detta är lovande med det _lilla_ dataset vi har :)


## Funderingar

[Mistral-7B-Instruct-v0.1-GGUF](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf?download=true)

```
➜  curl --output models/mistral-7b-instruct-v0.1.Q4_K_M.gguf https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf?download=true
```