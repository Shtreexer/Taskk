import spacy
import pandas as pd

nlp_custom = spacy.load('models/ner_mode')

#data = input('Write sm.. ')
data = ''
rez = []

while data != '/exit':
    data = input('Write sm.. ')
    doc = nlp_custom(data)#"Do elephants have excellent memory?"
    for ent in doc.ents:
        print(ent.text, ent.label_)

    rez.append([data, ent.label_])


pd.DataFrame(rez).iloc[:-1, :].to_csv('res/ner/test.csv')

