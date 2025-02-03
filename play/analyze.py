import spacy
import json
import re

file_path = "../data/SFB.json"

# Read the JSON data from the local file
with open(file_path, "r", encoding="utf-8") as file:
    json_data = json.load(file)

# pip install https://huggingface.co/KBLab/swedish-spacy-pipeline/resolve/main/swedish-spacy-pipeline-any-py3-none-any.whl
# nlp = spacy.load("swedish-spacy-pipeline")

# pip install https://github.com/explosion/spacy-models/releases/download/sv_core_news_lg-3.7.0/sv_core_news_lg-3.7.0-py3-none-any.whl
nlp = spacy.load('sv_core_news_lg')

# -- fixing lemma for certain words
lemmas = [
    ("avdelningarna", "avdelning"),
    ("adoptivföräldrarna", "adoptivföräldrar"),
    ("anpassningsbidraget", "anpassningsbidrag"),
    ("ansökningsmånaden", "ansökningsmånad"),
    ("ap-fonderna", "ap-fonder"),
    ("arbetsförhållandena", "arbetsförhållande"),
    ("arbetsförhållanden", "arbetsförhållande"),
    ("arbetslöshetskassor", "arbetslöshetskassa"),
    ("arbetsskadan", "arbetsskada"),
    ("arvsvinstfaktorer", "arvsvinstfaktor"),
    ("assistanstimmar", "assistanstimme"),
    ("assistanstimmarna", "assistanstimme"),
    ("karens", "karens"),
    ("penningböter", "penningbot")
]

attribute_ruler = nlp.get_pipe("attribute_ruler")
for a, b in lemmas:
    attribute_ruler.add([[{"TEXT": a}]], {"LEMMA": b})


ignoredTokens = [
    '.', '§', ')', '-', 'a.', 'a', 'alkohol', 'allmänna', 'alla', 'andra', 'angelägenhet',
    'annan', 'annat', 'antal', 'antalet', 'andning', 'anmält', 'anses', 'användning',
    'användningen', 'april', 'artikel', 'augusti', 'automatiserat', 'av', 'avbrottet',
    'avdelning', 'avdelningarna', 'avled', 'avresa', 'avsikt', 'avsikten', 'avse', 'avser',
    'avses', 'avslås', 'avstående', 'avståendet', 'avvikelse', 'avvikelser', 'b', 'b.',
    'd', 'de', 'den', 'dem', 'denne', 'denna', 'dessa', 'detsamma', 'det', 'detta', 'då', 'där',
    'därefter', 'e', 'efter', 'eg', 'en', 'ett', 'f', 'fall', 'faktorn', 'fattades', 'februari',
    'felet', 'femårsperiod', 'ferier', 'fjärdedel', 'fjärdedels', 'fråga', 'förd', 'färden',
    'följande', 'följd', 'för', 'förelåg', 'före', 'först', 'första', 'från', 'fram', 'frågan', 'gjorts',
    'gäller', 'gång', 'gången', 'gånger', 'göras', 'han', 'hans', 'hela', 'helt', 'hel', 'hennes',
    'henne', 'hon', 'honom', 'har', 'i', 'inleddes', 'intresse', 'ingen', 'kap', 'kap.', 'kr',
    'kronor', 'kortare', 'kvar', 'legat', 'lämnas', 'längst', 'm.fl.', 'med', 'män', 'någon',
    'något', 'nio', 'nr', 'nytt', 'om', 'ordning', 'pengar', 'procent', 'räknat', 'senare', 'sig',
    'situationen', 'situations', 'sjuk', 'sjukt', 'ska', 'skedde', 'skillnaden', 'skriftligen',
    'skäligen', 'skötsel', 'solidariskt', 'som', 'sondmatning', 'stadigvarande', 'storlek',
    'storleken', 'strecksatsen', 'stycket', 'styckena', 'största', 'summan', 'syfte', 'säljer',
    'sätt', 'sådant', 'såvitt', 'tal', 'talet', 'till', 'tillbaka', 'tillfällen', 'tillfälligt',
    'tillgodoräknas', 'tillsyn', 'tillsynen', 'tillämpliga', 'tillämpligheten', 'timmar', 'timme',
    'tjugofjärde', 'tjänster', 'tolftedel', 'tolv', 'tre', 'tretton', 'tusental', 'underlåten',
    'unga', 'uppehåll', 'uppehälle', 'uppenbarligen', 'uppfyllt', 'upphävda', 'upphörande',
    'upphör', 'uppkommer', 'upplupen', 'upplysa', 'uppnådd', 'uppnår', 'uppnås', 'uppnått',
    'uteslutande', 'utförandet', 'utsträckning', 'vad', 'valt', 'var', 'vars', 'vart', 'vecka',
    'veckan', 'veckor', 'vikt', 'vilande', 'vilket', 'vi', 'vissa', 'vistas', 'vistelse',
    'vistelsen', 'vuxen', 'värdet', 'väsentligen', 'vårdat', 'vården', 'vård', 'yngsta',
    'ändrades', 'ändringar', 'ändringen', 'ändring', 'är', 'åren', 'året', 'år', 'återinsjuknande',
    'åtskild', 'åttondels', 'ökningen', 'ökning', 'öre', 'överklaga', 'överklagar', 'överstiger',
    'övervägs'
]

rangePattern = r'^\d+\-\d+$'
rangePattern2 = r'^[a-z]\-\d+$'
rangePattern3 = r'^[a-z]\-[a-z]+$'
decimalPattern = r'^\d+,\d+$'
decimalPattern2 = r'^\d+\.\d+$'
numberPattern = r'^\d+$'
lawPattern = r'^\d{4}:\d+$'


def extract_texts(json_data):
    texts = []

    def traverse(obj, context):
        if isinstance(obj, dict):
            current_context = context.copy()
            if "nummer" in obj:
                if "kapitel" in context:
                    current_context["paragraf"] = obj["nummer"]
                else:
                    current_context["kapitel"] = obj["nummer"]
            if "namn" in obj:
                current_context["namn"] = obj["namn"]
            if "avdelning" in obj:
                current_context["avdelning_id"] = obj["avdelning"].get("id", "")
                current_context["avdelning_namn"] = obj["avdelning"].get("namn", "")
            if "underavdelning" in obj:
                current_context["underavdelning_id"] = obj["underavdelning"].get("id", "")
                current_context["underavdelning_namn"] = obj["underavdelning"].get("namn", "")
            if "rubrik" in obj:
                current_context["rubrik"] = obj["rubrik"]
            if "periodisering" in obj:
                if "kapitel" in current_context and "paragraf" not in current_context:
                    current_context["kapitel_periodisering"] = obj["periodisering"]
                if "paragraf" in current_context:
                    current_context["paragraf_periodisering"] = obj["periodisering"]
            if "stycke" in obj:
                for stycke in obj["stycke"]:
                    if isinstance(stycke, dict):
                        stycke_context = current_context.copy()
                        stycke_context["stycke"] = stycke["nummer"]
                        concatenated_text = "\n".join(stycke["text"])
                        texts.append({"context": stycke_context, "text": concatenated_text})
            for key, value in obj.items():
                traverse(value, current_context)
        elif isinstance(obj, list):
            for item in obj:
                traverse(item, context)

    traverse(json_data, {})
    return texts

def keyphrase(doc):
    for t in doc:
        if t.dep_ == 'pobj' and (t.pos_ == 'NOUN' or t.pos_ == 'PROPN'):
            return "/pobj/ " + (' '.join([child.text for child in t.lefts]) + ' ' + t.text).lstrip()

    for t in reversed(doc):
        if t.dep_ == 'nsubj' and (t.pos_ == 'NOUN' or t.pos_ == 'PROPN'):
            return "/nsubj/ " + t.text + ' ' + t.head.text

    for t in reversed(doc):
        if t.dep_ == 'dobj' and (t.pos_ == 'NOUN' or t.pos_ == 'PROPN'):
            return "/dobj/ " + t.head.text + ' ' + t.text

    return False

def extract_entities_and_relations(item, progress):
    entities = []
    relations = []
    rules = []

    context = item["context"]
    kapitel_info = context.get('kapitel')
    kapitel_periodisering = context.get('kapitel_periodisering') # optional
    paragraf_info = context.get('paragraf')
    paragraf_periodisering = context.get('paragraf_periodisering') # optional
    rubrik = context.get('rubrik') # optional

    doc = nlp(item["text"])
    progress.write(f"keyphrase: {keyphrase(doc)}\n")

    for token in doc:
        progress.write(token.text + " [" + token.dep_ + "] ")

        tokenText = token.text.lower()
        if tokenText not in ignoredTokens:
            if not (re.match(numberPattern, tokenText)
                    or re.match(decimalPattern, tokenText)   # 1,1234
                    or re.match(decimalPattern2, tokenText)  # 1.2
                    or re.match(rangePattern, tokenText)     # 1-100
                    or re.match(rangePattern2, tokenText)    # a-100
                    or re.match(rangePattern3, tokenText)):  # b-g
                # Identify noun phrases and named entities as potential entities
                if token.pos_ in ["NOUN", "PROPN"]:
                    print("entity: ", token.text)
                    entities.append((tokenText, token.lemma_, token.pos_))

                # if token.dep_ in ['nsubj', 'dobj', 'pobj', 'nmod']: # not 'nsubj:pass'
                #    (tokenText, token.lemma_, token.dep_, token.head.text.lower())

                # Extract relationships
                if token.dep_ in ["nsubj", "dobj"]:
                    subject = None
                    if token.dep_ == "nsubj":
                        subject = token
                        print("subject: ", token.text)
                    for child in token.head.children:
                        print("   child: ", child, child.dep_)
                    objects = [child for child in token.head.children if child.dep_ in ["dobj", "pobj"]]
                    for obj in objects:
                        print("object: ", obj)
                        relations.append((subject.text if subject else token.head.text, token.head.lemma_, obj.text))

                # Extract rules
                #if token.dep_ == "ROOT" or token.pos_ == "AUX":
                #    rules.append((token.text, token.lemma_, token.head.text, token.dep_))

    progress.write("\n")
    for entity in entities:
        progress.write("Entity: " + entity[0] + "\n")
    for relation in relations:
        progress.write("Relation: " + relation[0] + " " + relation[1] + " " + relation[2] + "\n")
    for rule in rules:
        progress.write("Rule: " + rule[0] + " " + rule[1] + " " + rule[2] + " " + rule[3] + "\n")
    progress.write("\n")

    return entities, relations, rules


def process_text(texts_with_context, progress):
    all_entities = []
    all_relations = []
    all_rules = []

    for item in texts_with_context:
        entities, relations, rules = extract_entities_and_relations(item, progress)
        all_entities.extend(entities)
        all_relations.extend(relations)
        all_rules.extend(rules)

    return all_entities, all_relations, all_rules


# Extract the texts with contextual information
print("Extracting text...")
texts_with_context = extract_texts(json_data)

# Process the entire JSON data
print("Processing text...")
with open("progress.txt", "w", encoding="utf-8") as progress:
    entities, relations, rules = process_text(texts_with_context, progress)

    sorted_entities = sorted(set(entities), key=lambda x: x[0] + x[1] + x[2])
    sorted_relations = sorted(set(relations), key=lambda x: x[0] + x[1] + x[2])
    sorted_rules = sorted(set(rules), key=lambda x: x[0] + x[1] + x[2])

    print("-"*80)
    print("- Entities:")
    print("-"*80)
    for entity in sorted_entities:
        print(entity)
    print()

    print("-"*80)
    print("- Relations:")
    print("-"*80)
    for relation in sorted_relations:
        print(relation)

    print("-"*80)
    print("- Rules:")
    print("-"*80)
    for relation in sorted_relations:
        print(relation)

# with open("analysis.txt", "w", encoding="utf-8") as f:
#     f.write("Entities: \n")
#     for entity in entities:
#         f.write("" + entity)
#         f.write("\n")
#
#     f.write("\n\nRelations: \n")
#     for relation in relations:
#         f.write("" + relation)
#         f.write("\n")

