from sentence_transformers import SentenceTransformer

model = SentenceTransformer("KBLab/sentence-bert-swedish-cased")


def embed(data):
    texts = []
    embeddings = []

    #     "lag": "Socialförsäkringsbalk (2010:110)",
    #     "avdelning": "B FAMILJEFÖRMÅNER",
    #     "underavdelning": "II Graviditetspenning och föräldrapenningsförmåner",
    #     "kapitel": "12",
    #     "kapitel_namn": "Föräldrapenning",
    #     "paragraf_rubrik": "Rätten till föräldrapenning",
    #     "paragraf_underrubrik": "Allmänna bestämmelser",
    #     "paragraf": "3",
    #     "referens": "Lag (2013:999)",
    #     "stycke": 1,
    #     "text": "För rätt till föräldrapenning..."

    for record in data:
        #
        #input_text = f"[AVDELNING {record['avdelning']}]"
        #input_text = f"{input_text} [UNDERAVDELNING {record['underavdelning']}]"

        #kapitel_namn = record["kapitel_namn"]
        #if kapitel_namn:
        #    input_text = f"{input_text} [KAPITEL {kapitel_namn}]"

        #paragraf_rubrik = record.get("paragraf_rubrik", None)
        #if paragraf_rubrik:
        #    input_text = f"{input_text} [RUBRIK {paragraf_rubrik}]"

        #paragraf_underrubrik = record.get("paragraf_underrubrik", None)
        #if paragraf_underrubrik:
        #    input_text = f"{input_text} [UNDERRUBRIK {paragraf_underrubrik}]"

        #input_text = f"{input_text} {record['text']}"

        input_text = record['text']
        embedding = model.encode(input_text, convert_to_numpy=True)

        # OBS:
        # model.encode() returns a single 1D NumPy array of shape (384,) when passed a single string
        #

        texts.append(input_text)
        embeddings.append(embedding)  # shape (384,)

    return texts, embeddings