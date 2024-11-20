# search_app.py

import streamlit as st
from py2neo import Graph
import pandas as pd
from rapidfuzz import process, fuzz
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import re

# Inisialisasi tokenizer dan model untuk NER
tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
model = AutoModelForTokenClassification.from_pretrained("indobenchmark/indobert-base-p1")
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Dictionary untuk normalisasi penyakit
disease_normalization = {
    "demam": "demam",
    "fever": "demam",
    "sakit kepala": "sakit kepala",
    "kepala pusing": "sakit kepala",
    "maag": "maag",
    "migrain": "migraine",  # Menangani typo 'migrain'
    "migraine": "migraine",
    "batuk": "batuk",
    "pilek": "pilek",
    "pusing": "pusing",
    "asma": "asma",
    "diare": "diare",
    "tipes": "tipes",
    # Tambahkan sinonim lainnya sesuai kebutuhan
}

# Fungsi untuk mengekstrak entitas penyakit dari query menggunakan Transformers
def extract_diseases(text):
    doc = ner_pipeline(text.lower())
    diseases = [ent['word'] for ent in doc if ent['entity_group'] in ['DISEASE', 'CONDITION', 'SYMPTOM']]
    if not diseases:
        # Fallback: gunakan pemisahan berbasis kata kunci
        keywords = ["batuk", "pilek", "demam", "sakit kepala", "pusing", "asma", "diare", "tipes", "maag", "migraine", "migrain"]
        diseases = [keyword for keyword in keywords if keyword in text.lower()]
    # Normalisasi nama penyakit
    normalized_diseases = [disease_normalization.get(d, d) for d in diseases]
    print(f"Extracted Diseases from '{text}': {normalized_diseases}")  # Debugging
    return normalized_diseases

# Koneksi ke Neo4j
graph = Graph("bolt://localhost:7687", auth=("neo4j", "Attaqy81"))

# Fungsi untuk mendapatkan semua nama obat untuk fuzzy matching dan auto-complete
@st.cache_data
def get_all_drug_names():
    query = "MATCH (d:Drug) RETURN d.name AS name"
    result = graph.run(query)
    drug_names = [record['name'] for record in result]
    print(f"All Drug Names Retrieved: {drug_names}")  # Debugging
    return drug_names

# Fungsi untuk mencari obat berdasarkan penyakit
def search_drugs_by_disease(disease):
    query = """
    MATCH (d:Drug)-[:treats]->(dis:Disease {name: $disease})
    RETURN d.name AS drug_name
    """
    result = graph.run(query, disease=disease)
    drugs = [record['drug_name'] for record in result]
    print(f"Drugs Treating '{disease}': {drugs}")  # Debugging
    return drugs

# Fungsi untuk mendapatkan atribut spesifik
def get_drug_price(drug_name):
    query = """
    MATCH (d:Drug {name: $drug_name})
    RETURN d.min_price AS MinPrice, d.max_price AS MaxPrice
    """
    result = graph.run(query, drug_name=drug_name)
    price_df = pd.DataFrame([record for record in result])
    print(f"Price DataFrame for '{drug_name}':\n{price_df}")  # Debugging
    return price_df

def get_drug_composition(drug_name):
    query = """
    MATCH (d:Drug)-[:contains]->(n:Ingredients)
    WHERE toLower(d.name) = toLower($drug_name)
    RETURN n.content AS Composition
    """
    result = graph.run(query, drug_name=drug_name)
    composition_df = pd.DataFrame([record for record in result])
    print(f"Composition DataFrame for '{drug_name}':\n{composition_df}")  # Debugging
    return composition_df

def get_drug_side_effects(drug_name):
    query = """
    MATCH (d:Drug)-[:hasSideEffect]->(n:SideEffect)
    WHERE toLower(d.name) = toLower($drug_name)
    RETURN n.effect AS SideEffects
    """
    result = graph.run(query, drug_name=drug_name)
    side_effects_df = pd.DataFrame([record for record in result])
    print(f"SideEffects DataFrame for '{drug_name}':\n{side_effects_df}")  # Debugging
    return side_effects_df

def get_drug_manufacturer(drug_name):
    query = """
    MATCH (d:Drug)-[:producedBy]->(n:Manufacturer)
    WHERE toLower(d.name) = toLower($drug_name)
    RETURN n.name AS Manufacturer
    """
    result = graph.run(query, drug_name=drug_name)
    manufacturer_df = pd.DataFrame([record for record in result])
    print(f"Manufacturer DataFrame for '{drug_name}':\n{manufacturer_df}")  # Debugging
    return manufacturer_df

def get_drug_precautions(drug_name):
    query = """
    MATCH (d:Drug)-[:requiresPrecaution]->(n:Precaution)
    WHERE toLower(d.name) = toLower($drug_name)
    RETURN n.details AS Precautions
    """
    result = graph.run(query, drug_name=drug_name)
    precautions_df = pd.DataFrame([record for record in result])
    print(f"Precautions DataFrame for '{drug_name}':\n{precautions_df}")  # Debugging
    return precautions_df

def get_drug_description(drug_name):
    query = """
    MATCH (d:Drug)
    WHERE toLower(d.name) = toLower($drug_name)
    RETURN d.description AS Description
    """
    result = graph.run(query, drug_name=drug_name)
    description_df = pd.DataFrame([record for record in result])
    print(f"Description DataFrame for '{drug_name}':\n{description_df}")  # Debugging
    return description_df

# Fungsi untuk mencari obat terkait
def get_related_drugs(drug_name, limit=5):
    query = """
    MATCH (d:Drug {name: $drug_name})-[:treats]->(dis:Disease)<-[:treats]-(related:Drug)
    WHERE related.name <> $drug_name
    RETURN DISTINCT related.name AS related_drug
    LIMIT $limit
    """
    result = graph.run(query, drug_name=drug_name, limit=limit)
    related_drugs = [record['related_drug'] for record in result]
    print(f"Related Drugs for '{drug_name}': {related_drugs}")  # Debugging
    return related_drugs

# Fungsi utama untuk mencari berdasarkan query pengguna
def search_query(query):
    diseases = extract_diseases(query)
    
    if diseases:
        # Pencarian obat berdasarkan penyakit
        drugs = set()
        for disease in diseases:
            matched_drugs = search_drugs_by_disease(disease)
            drugs.update(matched_drugs)
        
        if drugs:
            print(f"Drugs Retrieved for Diseases {diseases}: {drugs}")  # Debugging
            return pd.DataFrame({"Drugs": list(drugs)})
        else:
            print("Tidak ditemukan obat untuk penyakit yang diekstrak.")  # Debugging
            return pd.DataFrame()
    else:
        # Pencarian berdasarkan nama obat atau atribut lain
        all_drug_names = get_all_drug_names()
        matched_drug, score, _ = process.extractOne(
            query, all_drug_names, scorer=fuzz.WRatio
        )
        print(f"Matched Drug: {matched_drug} with score: {score}")  # Debugging
        
        if score >= 80:
            # Normalize drug name by removing dosage information for matching
            base_drug_name = re.sub(r'\s*\d+\s*(mg|ml|g)\s*\d+\s*tablet.*', '', matched_drug, flags=re.IGNORECASE).strip()
            matched_drug = base_drug_name if base_drug_name else matched_drug
            print(f"Base drug name after normalization: {matched_drug}")  # Debugging
            
            # Temukan semua varian obat yang cocok
            matched_drugs = [drug for drug in all_drug_names if drug.lower().startswith(matched_drug.lower())]
            print(f"Matched drugs after normalization: {matched_drugs}")  # Debugging
            
            if matched_drugs:
                print(f"Drugs Retrieved after normalization: {matched_drugs}")  # Debugging
                return pd.DataFrame({"Drugs": matched_drugs})
            else:
                print(f"Drugs Retrieved: {matched_drug}")  # Debugging
                return pd.DataFrame({"Drugs": [matched_drug]})
        else:
            print("Tidak ada kecocokan yang cukup baik.")  # Debugging
            return pd.DataFrame()

# Fungsi untuk memberikan beberapa saran
def get_multiple_suggestions(search_term, all_drug_names, limit=5):
    suggestions = process.extract(search_term, all_drug_names, scorer=fuzz.WRatio, limit=limit)
    # Filter saran yang relevan, misalnya hanya jika skor di atas threshold tertentu
    filtered_suggestions = [s for s in suggestions if s[1] >= 60]
    print(f"Suggestions for '{search_term}': {filtered_suggestions}")  # Debugging
    return filtered_suggestions

# Fungsi untuk menampilkan informasi berdasarkan tipe query
def display_information(suggestion, query):
    st.subheader(f"Informasi untuk {suggestion}")
    
    # Komposisi
    composition_df = get_drug_composition(suggestion)
    st.write("Composition DataFrame:", composition_df)  # Debugging
    if not composition_df.empty and 'Composition' in composition_df.columns:
        st.markdown("**Komposisi:**")
        st.write(composition_df.iloc[0]['Composition'])
    else:
        st.markdown("**Komposisi:**")
        st.write("Tidak ada informasi komposisi.")
    
    # Efek Samping
    side_effects_df = get_drug_side_effects(suggestion)
    st.write("SideEffects DataFrame:", side_effects_df)  # Debugging
    if not side_effects_df.empty and 'SideEffects' in side_effects_df.columns:
        st.markdown("**Efek Samping:**")
        st.write(", ".join(side_effects_df['SideEffects'].tolist()))
    else:
        st.markdown("**Efek Samping:**")
        st.write("Tidak ada informasi efek samping.")
    
    # Harga
    price_df = get_drug_price(suggestion)
    st.write("Price DataFrame:", price_df)  # Debugging
    if not price_df.empty and 'MinPrice' in price_df.columns and 'MaxPrice' in price_df.columns:
        min_price = price_df.iloc[0]['MinPrice']
        max_price = price_df.iloc[0]['MaxPrice']
        st.markdown("**Harga:**")
        st.write(f"Rp{min_price:,.2f} - Rp{max_price:,.2f}")
    else:
        st.markdown("**Harga:**")
        st.write("Tidak ada informasi harga.")
    
    # Manufacturer
    manufacturer_df = get_drug_manufacturer(suggestion)
    st.write("Manufacturer DataFrame:", manufacturer_df)  # Debugging
    if not manufacturer_df.empty and 'Manufacturer' in manufacturer_df.columns:
        st.markdown("**Manufaktur:**")
        st.write(manufacturer_df.iloc[0]['Manufacturer'])
    else:
        st.markdown("**Manufaktur:**")
        st.write("Tidak ada informasi manufaktur.")
    
    # Precautions
    precautions_df = get_drug_precautions(suggestion)
    st.write("Precautions DataFrame:", precautions_df)  # Debugging
    if not precautions_df.empty and 'Precautions' in precautions_df.columns:
        st.markdown("**Perhatian:**")
        st.write(precautions_df.iloc[0]['Precautions'])
    else:
        st.markdown("**Perhatian:**")
        st.write("Tidak ada informasi perhatian.")
    
    # Deskripsi
    description_df = get_drug_description(suggestion)
    st.write("Description DataFrame:", description_df)  # Debugging
    if not description_df.empty and 'Description' in description_df.columns:
        st.markdown("**Deskripsi:**")
        st.write(description_df.iloc[0]['Description'])
    else:
        st.markdown("**Deskripsi:**")
        st.write("Tidak ada informasi deskripsi.")
    
    # Obat Terkait
    related_drugs = get_related_drugs(suggestion)
    if related_drugs:
        st.markdown("**Obat Terkait:**")
        for related in related_drugs:
            if st.button(related):
                display_information(related, query)

# Streamlit app layout
st.title("Sistem Informasi Obat Berbasis Knowledge Graph")
st.write("Cari informasi tentang obat menggunakan knowledge graph.")

# Input query pencarian
search_term = st.text_input("Masukkan query Anda (misalnya, 'obat untuk batuk', 'harga paramex', 'obat untuk asma')")

if search_term:
    st.write(f"Mencari informasi terkait: **{search_term}**")
    
    # Ekstraksi entitas penyakit dari query
    diseases = extract_diseases(search_term)
    
    if diseases:
        # Jika ada penyakit yang diekstrak, lakukan pencarian berdasarkan penyakit
        search_results_df = search_query(search_term)
        
        if not search_results_df.empty:
            st.write("**Obat yang mengobati kondisi yang ditentukan:**")
            for index, row in search_results_df.iterrows():
                if st.button(row['Drugs']):
                    display_information(row['Drugs'], search_term)
        else:
            st.write("Tidak ditemukan informasi untuk query dan penyakit yang ditentukan.")
            
            # Berikan beberapa saran alternatif
            all_drug_names = get_all_drug_names()
            suggestions = get_multiple_suggestions(search_term, all_drug_names)
            
            if suggestions:
                st.write("**Mungkin Anda maksud:**")
                for suggestion, score, _ in suggestions:
                    # Buat tombol untuk setiap saran
                    if st.button(f"{suggestion} (Confidence: {score}%)"):
                        # Tampilkan informasi obat
                        display_information(suggestion, search_term)
    else:
        # Jika tidak ada penyakit yang diekstrak, coba pencarian berdasarkan nama obat atau atribut lain
        search_results_df = search_query(search_term)
        
        if not search_results_df.empty:
            if "Drugs" in search_results_df.columns:
                st.write("**Informasi Obat:**")
                for index, row in search_results_df.iterrows():
                    if st.button(row['Drugs']):
                        display_information(row['Drugs'], search_term)
            else:
                # Tampilkan atribut spesifik
                if "MinPrice" in search_results_df.columns and "MaxPrice" in search_results_df.columns:
                    st.write("**Informasi Harga:**")
                    for index, row in search_results_df.iterrows():
                        st.write(f"Rp{row['MinPrice']:,.2f} - Rp{row['MaxPrice']:,.2f}")
                elif "Composition" in search_results_df.columns:
                    st.write("**Informasi Komposisi:**")
                    for index, row in search_results_df.iterrows():
                        st.write(row['Composition'])
                elif "SideEffects" in search_results_df.columns:
                    st.write("**Informasi Efek Samping:**")
                    for index, row in search_results_df.iterrows():
                        st.write(row['SideEffects'])
                elif "Manufacturer" in search_results_df.columns:
                    st.write("**Informasi Manufaktur:**")
                    for index, row in search_results_df.iterrows():
                        st.write(row['Manufacturer'])
                elif "Precautions" in search_results_df.columns:
                    st.write("**Informasi Perhatian:**")
                    for index, row in search_results_df.iterrows():
                        st.write(row['Precautions'])
                else:
                    st.write("**Informasi:**")
                    for index, row in search_results_df.iterrows():
                        st.write(row['Drugs'])
        else:
            st.write("Tidak ditemukan informasi untuk query yang ditentukan.")
            
            # Berikan beberapa saran alternatif
            all_drug_names = get_all_drug_names()
            suggestions = get_multiple_suggestions(search_term, all_drug_names)
            
            if suggestions:
                st.write("**Mungkin Anda maksud:**")
                for suggestion, score, _ in suggestions:
                    # Buat tombol untuk setiap saran
                    if st.button(f"{suggestion} (Confidence: {score}%)"):
                        # Tampilkan informasi berdasarkan jenis query
                        display_information(suggestion, search_term)
