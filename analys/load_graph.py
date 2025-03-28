import json
from neo4j import GraphDatabase

file_path = "../sfbreader/data/SFB-flat.json"

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
      - (k:Kapitel {number: ..., namn:...})
      - (p:Paragraf {number: ...})
      - (p:Stycke {number: ..., text: ...})
    Then link them with relationships.
    """

    # Build string IDs
    kapitel_id = f"K{record['kapitel']}"
    paragraf_id = f"{kapitel_id}P{record['paragraf']}"
    stycke_id = f"{paragraf_id}S{record['stycke']}"

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
          id: $kapitelId, 
          nummer: $kapitelNummer, 
          namn: $kapitelNamn, 
          underavdelning: $underavdelning,
          avdelning: $avdelning,
          lag: $lagNamn
        })
        MERGE (u)-[:HAR_KAPITEL]->(k)
        
        MERGE (p:Paragraf {
          id: $paragrafId, 
          nummer: $paragrafNummer,
          kapitel: $kapitelId,
          underavdelning: $underavdelning,
          avdelning: $avdelning,
          lag: $lagNamn
        })
        MERGE (k)-[:HAR_PARAGRAF]->(p)
        
        MERGE (s:Stycke {
          id: $styckeId, 
          nummer: $styckeNummer,
          kategori: $kategori,
          paragraf: $paragrafId,
          kapitel: $kapitelId,
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
        kapitelId=kapitel_id,
        kapitelNummer=record["kapitel"],
        kapitelNamn=record["kapitel_namn"],

        # For the Paragraf node
        paragrafId=paragraf_id,
        paragrafNummer=record["paragraf"],

        # For the Stycke node
        kategori=record.get("kategori", ""),
        styckeId=stycke_id,
        styckeNummer=record["stycke"],

        text=record["text"]
    )

def link_next(tx):
    tx.run("""
        MATCH (p:Paragraf)-[:HAR_STYCKE]->(s)
        WITH p, s
        ORDER BY p.nummer, s.nummer  // adjust property names if needed
        WITH collect(s) AS stycken
        UNWIND range(0, size(stycken)-2) AS i
        WITH stycken[i] AS s1, stycken[i+1] AS s2
        RETURN id(s1) AS source, id(s2) AS target, 'NEXT' AS type
    """)

# Insert data into Neo4j
with driver.session() as session:
    for record in data:
        session.execute_write(create_graph, record)
    session.execute_write(link_next)

driver.close()
