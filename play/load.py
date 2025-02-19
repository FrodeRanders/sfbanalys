from neo4j import GraphDatabase
import json

file_path = "../../sfbreader/data/SFB-flat.json"

# Read the JSON data from the local file
with open(file_path, "r", encoding="utf-8") as file:
    data = json.load(file)

uri = "bolt://localhost:7687"
user = "neo4j"
password = "sfbsfbsfb"
driver = GraphDatabase.driver(uri, auth=(user, password))


def create_graph(tx, record):
    """
    For a given law record, create or match:
      - (l:Lag {name: ...})
      - (a:Avdelning {name: ...})
      - (u:Underavdelning {name: ...})
      - (k:Kapitel {number: ...})
      - (p:Paragraf {number: ...})
      - (p:Stycke {number: ..., text: ...})
    Then link them with relationships.
    """

    # Build string IDs
    kapitel_namn = f"K{record['kapitel']}"
    paragraf_namn = f"{kapitel_namn}P{record['paragraf']}"
    stycke_namn = f"{paragraf_namn}S{record['stycke']}"

    query = """
        MERGE (l:Lag { namn: $lagNamn })
        MERGE (a:Avdelning { namn: $avdelning, lag: $lagNamn })
        MERGE (l)-[:HAR_AVDELNING]->(a)
        
        MERGE (u:Underavdelning { 
          namn: $underavdelning, 
          avdelning: $avdelning, 
          lag: $lagNamn 
        })
        MERGE (a)-[:HAR_UNDERAVDELNING]->(u)
            
        MERGE (k:Kapitel { 
          namn: $kapitelNamn, 
          nummer: $kapitelNummer, 
          underavdelning: $underavdelning,
          avdelning: $avdelning,
          lag: $lagNamn
        })
        MERGE (u)-[:HAR_KAPITEL]->(k)
        
        MERGE (p:Paragraf {
          namn: $paragrafNamn, 
          nummer: $paragrafNummer,
          kapitel: $kapitelNamn,
          underavdelning: $underavdelning,
          avdelning: $avdelning,
          lag: $lagNamn
        })
        MERGE (k)-[:HAR_PARAGRAF]->(p)
        
        MERGE (s:Stycke {
          namn: $styckeNamn, 
          nummer: $styckeNummer,
          paragraf: $paragrafNamn,
          kapitel: $kapitelNamn,
          underavdelning: $underavdelning,
          avdelning: $avdelning,
          lag: $lagNamn
        })
        SET s.text = $text
        MERGE (p)-[:HAR_STYCKE]->(s)
    """
    tx.run(
        query,
        lagNamn=record["lag"],
        avdelning=record["avdelning"],
        underavdelning=record["underavdelning"],

        # For the Kapitel node
        kapitelNamn=kapitel_namn,
        kapitelNummer=record["kapitel"],

        # For the Paragraf node
        paragrafNamn=paragraf_namn,
        paragrafNummer=record["paragraf"],

        # For the Stycke node
        styckeNamn=stycke_namn,
        styckeNummer=record["stycke"],

        text=record["text"]
    )


# Insert data into Neo4j
with driver.session() as session:
    for record in data:
        session.execute_write(create_graph, record)

driver.close()
