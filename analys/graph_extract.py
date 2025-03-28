from neo4j import GraphDatabase
import pandas as pd

# === Configure your connection ===
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "sfbsfbsfb"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def get_stycke_nodes(tx):
    query = """
    MATCH (k:Kapitel)-[:HAR_PARAGRAF]->(p:Paragraf)-[:HAR_STYCKE]->(s:Stycke)
    RETURN id(s) AS id, 
           s.avdelning AS avdelning,
           s.underavdelning AS underavdelning,
           k.nummer AS kapitelNummer,
           k.namn AS kapitelNamn,
           p.nummer AS paragrafNummer,
           s.kategori AS kategori,
           s.text AS text
    """
    return tx.run(query).data()

def get_edges_same_paragraf(tx):
    query = """
    MATCH (p:Paragraf)-[:HAR_STYCKE]->(s1:Stycke),
          (p)-[:HAR_STYCKE]->(s2:Stycke)
    WHERE id(s1) < id(s2)
    RETURN id(s1) AS source, id(s2) AS target, 'IN_PARAGRAF' AS type
    """
    return tx.run(query).data()

def get_edges_same_kapitel(tx):
    query = """
    MATCH (k:Kapitel)-[:HAR_PARAGRAF]->(p1:Paragraf)-[:HAR_STYCKE]->(s1:Stycke),
          (k)-[:HAR_PARAGRAF]->(p2:Paragraf)-[:HAR_STYCKE]->(s2:Stycke)
    WHERE id(s1) < id(s2)
    RETURN id(s1) AS source, id(s2) AS target, 'IN_KAPITEL' AS type
    """
    return tx.run(query).data()

def run():
    with driver.session() as session:
        print("Extracting nodes...")
        nodes = session.execute_read(get_stycke_nodes)
        df_nodes = pd.DataFrame(nodes)
        df_nodes.to_csv("nodes.csv", index=False)
        print(f"Wrote {len(df_nodes)} nodes to nodes.csv")

        print("Extracting edges...")
        edges_paragraf = session.execute_read(get_edges_same_paragraf)
        edges_kapitel = session.execute_read(get_edges_same_kapitel)
        df_edges = pd.DataFrame(edges_paragraf + edges_kapitel)
        df_edges.to_csv("edges.csv", index=False)
        print(f"Wrote {len(df_edges)} edges to edges.csv")

if __name__ == "__main__":
    run()
