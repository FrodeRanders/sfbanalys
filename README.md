# sfbanalys
Analys av Socialförsäkringsbalken.

## Setup
Såhär riggar du Neo4j för körning:
```
➜  docker pull neo4j            
➜  docker run -d --name neo4j -p7474:7474 -p7687:7687 -e NEO4J_AUTH=neo4j/sfbsfbsfb -v ./data:/data neo4j
```

Såhär riggar du Python för körning:
```
➜  python3.12 -m venv env
➜  source env/bin/activate
➜  python3 -m pip install -r requirements.txt
```

## Köra

```
➜  python3 analys/load_graph.py
➜  python3 analys/graph_extract.py
➜  python3 analys/classify.py

```
