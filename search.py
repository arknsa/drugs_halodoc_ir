# search_app.py

import streamlit as st
from py2neo import Graph
import pandas as pd
from rapidfuzz import process, fuzz
import spacy
import re

# Load spaCy model for NER (gunakan model multibahasa atau model khusus Bahasa Indonesia jika tersedia)
try:
    nlp = spacy.load("xx_ent_wiki_sm")
except OSError:
    st.error("Model 'xx_ent_wiki_sm' tidak ditemukan. Silakan instal model tersebut dengan menjalankan:\npython -m spacy download xx_ent_wiki_sm")
    st.stop()

# Tambahkan dictionary untuk normalisasi penyakit
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

# Fungsi untuk mengekstrak entitas penyakit dari query
def extract_diseases(text):
    doc = nlp(text)
    diseases = [ent.text.strip().lower() for ent in doc.ents if ent.label_ in ['DISEASE', 'CONDITION', 'SYMPTOM']]
    if not diseases:
        # Fallback: gunakan pemisahan berbasis kata kunci
        keywords = ["batuk", "pilek", "demam", "sakit kepala", "pusing", "asma", "diare", "tipes", "maag", "migraine", "migrain"]
        diseases = [keyword for keyword in keywords if keyword in text.lower()]
    # Normalisasi nama penyakit
    normalized_diseases = [disease_normalization.get(d, d) for d in diseases]
    return normalized_diseases

# Fungsi untuk mengekstrak kondisi harga dari query
def extract_price_condition(query):
    # Contoh: "harga dibawah 10000", "harga di atas 50000", "harga antara 10000 dan 20000"
    pattern = r'(dibawah|di bawah|dibelakang|di atas|diatas|antara)\s*Rp?\.?(\d+\.?\d*)\s*(dan|hingga)?\s*Rp?\.?(\d+\.?\d*)?'
    match = re.search(pattern, query.lower())
    if match:
        condition = match.group(1)
        value1 = int(match.group(2).replace('.', '').replace(',', ''))
        value2 = match.group(4)
        if condition in ['dibawah', 'di bawah', 'dibelakang']:
            return '<', value1
        elif condition in ['di atas', 'diatas']:
            return '>', value1
        elif condition == 'antara' and value2:
            return 'between', (value1, int(value2.replace('.', '').replace(',', '')))
    return None, None

# Fungsi untuk parse harga
def parse_price(price_str):
    # Remove 'Rp', '.', ',', and convert to integers
    try:
        price_str = price_str.replace('Rp', '').replace('.', '').replace(',', '').strip()
        parts = price_str.split('-')
        if len(parts) == 2:
            min_price = int(parts[0].strip())
            max_price = int(parts[1].strip())
            return min_price, max_price
        elif len(parts) == 1:
            # Single price
            price = int(parts[0].strip())
            return price, price
        else:
            return None, None
    except:
        return None, None

# Koneksi ke Neo4j database
graph = Graph("bolt://localhost:7687", auth=("neo4j", "Attaqy81"))

# Fungsi untuk mendapatkan semua nama obat untuk fuzzy matching dan auto-complete
@st.cache_data
def get_all_drug_names():
    query = "MATCH (d:Drug) RETURN d.name AS name"
    result = graph.run(query)
    return [record['name'] for record in result]

# Fungsi untuk mendapatkan semua nama penyakit
@st.cache_data
def get_all_diseases():
    query = "MATCH (d:Disease) RETURN d.name AS name"
    result = graph.run(query)
    return [record['name'] for record in result]

# Fungsi untuk mendapatkan semua nama precaution
@st.cache_data
def get_all_precautions():
    query = "MATCH (p:Precautions) RETURN p.details AS detail"
    result = graph.run(query)
    return [record['detail'] for record in result]

# Fungsi untuk mencari obat berdasarkan penyakit
def search_drugs_by_disease(disease):
    query = """
    MATCH (d:Drug)-[:treats]->(dis:Disease {name: $disease})
    RETURN d.name AS drug_name
    """
    result = graph.run(query, disease=disease)
    return [record['drug_name'] for record in result]

# Fungsi untuk mencari obat berdasarkan precaution
def search_drugs_by_precaution(precaution):
    query = """
    MATCH (d:Drug)-[:requiresPrecaution]->(p:Precautions {details: $precaution})
    RETURN d.name AS drug_name
    """
    result = graph.run(query, precaution=precaution)
    return [record['drug_name'] for record in result]

# Fungsi untuk mendapatkan atribut spesifik
def get_drug_price(drug_name):
    query = """
    MATCH (d:Drug {name: $drug_name})
    RETURN d.price AS Price
    """
    result = graph.run(query, drug_name=drug_name)
    return pd.DataFrame([record for record in result])

def get_drug_composition(drug_name):
    query = """
    MATCH (d:Drug {name: $drug_name})-[:contains]->(n:Ingredients)
    RETURN n.content AS Composition
    """
    result = graph.run(query, drug_name=drug_name)
    return pd.DataFrame([record for record in result])

def get_drug_side_effects(drug_name):
    query = """
    MATCH (d:Drug {name: $drug_name})-[:hasSideEffect]->(n:SideEffects)
    RETURN n.effects AS SideEffects
    """
    result = graph.run(query, drug_name=drug_name)
    return pd.DataFrame([record for record in result])

def get_drug_manufacturer(drug_name):
    query = """
    MATCH (d:Drug {name: $drug_name})-[:producedBy]->(n:Manufacturer)
    RETURN n.name AS Manufacturer
    """
    result = graph.run(query, drug_name=drug_name)
    return pd.DataFrame([record for record in result])

def get_drug_precautions(drug_name):
    query = """
    MATCH (d:Drug {name: $drug_name})-[:requiresPrecaution]->(n:Precautions)
    RETURN n.details AS Precautions
    """
    result = graph.run(query, drug_name=drug_name)
    return pd.DataFrame([record for record in result])

# Fungsi untuk mencari informasi umum obat
def search_drug_information(drug_name, query_type):
    if query_type == "General Information":
        query = """
        MATCH (d:Drug {name: $drug_name})-[r]->(n)
        RETURN type(r) AS relationship, n
        """
    
    elif query_type == "Side Effects":
        query = """
        MATCH (d:Drug {name: $drug_name})-[:hasSideEffect]->(n:SideEffects)
        RETURN n.effects AS SideEffects
        """
        
    elif query_type == "Dosage":
        query = """
        MATCH (d:Drug {name: $drug_name})
        RETURN d.dosage AS Dosage
        """
    
    elif query_type == "Manufacturer":
        query = """
        MATCH (d:Drug {name: $drug_name})-[:producedBy]->(n:Manufacturer)
        RETURN n.name AS Manufacturer
        """
    
    elif query_type == "Indications":
        query = """
        MATCH (d:Drug {name: $drug_name})-[:treats]->(n:Disease)
        RETURN n.name AS Indications
        """
    
    elif query_type == "Precautions":
        query = """
        MATCH (d:Drug {name: $drug_name})-[:requiresPrecaution]->(n:Precautions)
        RETURN n.details AS Precautions
        """
        
    result = graph.run(query, drug_name=drug_name)
    return pd.DataFrame([record for record in result])

# Fungsi utama untuk mencari berdasarkan query pengguna
def search_query(query):
    diseases = extract_diseases(query)
    drugs_treating = set()
    drugs_precaution = set()
    price_filtered_drugs = set()
    
    if diseases:
        for disease in diseases:
            matched_drugs = search_drugs_by_disease(disease)
            drugs_treating.update(matched_drugs)
    
    # Cari precaution jika query mencakup precaution
    all_precautions = get_all_precautions()
    matched_precaution, score, _ = process.extractOne(
        query, all_precautions, scorer=fuzz.WRatio
    )
    if score >= 60:
        drugs_precaution = set(search_drugs_by_precaution(matched_precaution))
    
    # Handle kondisi harga
    operator, value = extract_price_condition(query)
    
    if operator and value:
        all_drug_names = get_all_drug_names()
        for drug in all_drug_names:
            df_price = get_drug_price(drug)
            if not df_price.empty:
                price_str = df_price['Price'].iloc[0]
                min_price, max_price = parse_price(price_str)
                if min_price is None:
                    continue
                if operator == '<' and max_price < value:
                    price_filtered_drugs.add(drug)
                elif operator == '>' and min_price > value:
                    price_filtered_drugs.add(drug)
                elif operator == 'between' and min_price >= value[0] and max_price <= value[1]:
                    price_filtered_drugs.add(drug)
    
    return {
        "drugs_treating": list(drugs_treating),
        "drugs_precaution": list(drugs_precaution),
        "price_filtered_drugs": list(price_filtered_drugs)
    }

# Fungsi untuk memberikan beberapa saran
def get_multiple_suggestions(search_term, all_names, limit=3):
    suggestions = process.extract(search_term, all_names, scorer=fuzz.WRatio, limit=limit)
    # Filter saran yang relevan, misalnya hanya jika skor di atas threshold tertentu
    filtered_suggestions = [s for s in suggestions if s[1] >= 60]
    return filtered_suggestions

# Fungsi untuk menampilkan informasi berdasarkan tipe query
def display_information(suggestion, query):
    if "harga" in query.lower():
        search_results_df = get_drug_price(suggestion)
        if not search_results_df.empty:
            st.write(f"**Price** for **{suggestion}**:")
            st.dataframe(search_results_df)
    elif "komposisi" in query.lower():
        search_results_df = get_drug_composition(suggestion)
        if not search_results_df.empty:
            st.write(f"**Composition** of **{suggestion}**:")
            st.dataframe(search_results_df)
    elif "efek samping" in query.lower():
        search_results_df = get_drug_side_effects(suggestion)
        if not search_results_df.empty:
            st.write(f"**Side Effects** of **{suggestion}**:")
            st.dataframe(search_results_df)
    elif "manufaktur" in query.lower():
        search_results_df = get_drug_manufacturer(suggestion)
        if not search_results_df.empty:
            st.write(f"**Manufacturer** of **{suggestion}**:")
            st.dataframe(search_results_df)
    elif "perhatian" in query.lower():
        search_results_df = get_drug_precautions(suggestion)
        if not search_results_df.empty:
            st.write(f"**Precautions** for **{suggestion}**:")
            st.dataframe(search_results_df)
    else:
        # General Information jika jenis query tidak dikenali
        search_results_df = search_drug_information(suggestion, "General Information")
        if not search_results_df.empty:
            st.write(f"**General Information** for **{suggestion}**:")
            st.dataframe(search_results_df)
        else:
            st.write("No additional information found for the selected drug.")

# Streamlit app layout
st.title("Drug Information Retrieval System")
st.write("Search for information about drugs using the knowledge graph.")

# Search query input
search_term = st.text_input("Enter your query (e.g., 'obat untuk batuk', 'harga paramex', 'obat untuk asma')")

if search_term:
    st.write(f"Searching for information related to: **{search_term}**")
    
    # Ekstraksi entitas penyakit dari query
    diseases = extract_diseases(search_term)
    
    if diseases:
        # Jika ada penyakit yang diekstrak, lakukan pencarian berdasarkan penyakit
        search_results = search_query(search_term)
        
        drugs_treating = search_results.get("drugs_treating", [])
        drugs_precaution = search_results.get("drugs_precaution", [])
        drugs_price_condition = search_results.get("price_filtered_drugs", [])
        
        if drugs_treating:
            st.write("**Drugs that treat the specified condition:**")
            st.dataframe(pd.DataFrame({"Drugs": drugs_treating}))
        else:
            st.write("No drugs found that treat the specified condition.")
        
        if drugs_precaution:
            st.write("**Drugs that require precaution for the specified condition:**")
            st.dataframe(pd.DataFrame({"Drugs": drugs_precaution}))
        
        if drugs_price_condition:
            st.write("**Drugs that meet the price condition:**")
            st.dataframe(pd.DataFrame({"Drugs": drugs_price_condition}))
        else:
            # Jika ada kondisi harga tapi tidak ada obat yang memenuhi
            operator, value = extract_price_condition(search_term)
            if operator and value:
                if operator == '<':
                    st.write(f"No drugs found with price below Rp{value:,}")
                elif operator == '>':
                    st.write(f"No drugs found with price above Rp{value:,}")
                elif operator == 'between':
                    st.write(f"No drugs found with price between Rp{value[0]:,} and Rp{value[1]:,}")
    else:
        # Jika tidak ada penyakit yang diekstrak, mungkin query tentang harga atau atribut lain
        # Check if the query contains price conditions
        operator, value = extract_price_condition(search_term)
        
        if operator and value:
            # Jika ada kondisi harga, filter obat berdasarkan harga
            all_drug_names = get_all_drug_names()
            filtered_drugs = set()
            for drug in all_drug_names:
                df_price = get_drug_price(drug)
                if not df_price.empty:
                    price_str = df_price['Price'].iloc[0]
                    min_price, max_price = parse_price(price_str)
                    if min_price is None:
                        continue
                    if operator == '<' and max_price < value:
                        filtered_drugs.add(drug)
                    elif operator == '>' and min_price > value:
                        filtered_drugs.add(drug)
                    elif operator == 'between' and min_price >= value[0] and max_price <= value[1]:
                        filtered_drugs.add(drug)
            
            if filtered_drugs:
                if operator == '<':
                    st.write(f"**Drugs with price below Rp{value:,}:**")
                elif operator == '>':
                    st.write(f"**Drugs with price above Rp{value:,}:**")
                elif operator == 'between':
                    st.write(f"**Drugs with price between Rp{value[0]:,} and Rp{value[1]:,}:**")
                st.dataframe(pd.DataFrame({"Drugs": list(filtered_drugs)}))
            else:
                if operator == '<':
                    st.write(f"No drugs found with price below Rp{value:,}.")
                elif operator == '>':
                    st.write(f"No drugs found with price above Rp{value:,}.")
                elif operator == 'between':
                    st.write(f"No drugs found with price between Rp{value[0]:,} and Rp{value[1]:,}.")
        else:
            # Jika tidak ada kondisi harga, lakukan pencarian atribut lain seperti harga, komposisi, dll.
            search_results = search_query(search_term)
            
            drugs_treating = search_results.get("drugs_treating", [])
            drugs_precaution = search_results.get("drugs_precaution", [])
            drugs_price_condition = search_results.get("price_filtered_drugs", [])
            
            # Menampilkan informasi berdasarkan kata kunci
            if "harga" in search_term.lower():
                st.write("**Price Information:**")
            elif "komposisi" in search_term.lower():
                st.write("**Composition Information:**")
            elif "efek samping" in search_term.lower():
                st.write("**Side Effects Information:**")
            elif "manufaktur" in search_term.lower():
                st.write("**Manufacturer Information:**")
            elif "perhatian" in search_term.lower():
                st.write("**Precautions Information:**")
            else:
                st.write("**Information:**")
            
            # Menampilkan DataFrame jika ada
            if drugs_treating or drugs_precaution or drugs_price_condition:
                if drugs_treating:
                    st.write("**Drugs that treat the specified condition:**")
                    st.dataframe(pd.DataFrame({"Drugs": drugs_treating}))
                if drugs_precaution:
                    st.write("**Drugs that require precaution for the specified condition:**")
                    st.dataframe(pd.DataFrame({"Drugs": drugs_precaution}))
                if drugs_price_condition:
                    st.write("**Drugs that meet the price condition:**")
                    st.dataframe(pd.DataFrame({"Drugs": drugs_price_condition}))
            else:
                st.write("No information found for the specified query.")
                
                # Berikan beberapa saran alternatif
                all_drug_names = get_all_drug_names()
                suggestions = get_multiple_suggestions(search_term, all_drug_names, limit=3)
                
                if suggestions:
                    st.write("**Did you mean:**")
                    for suggestion, score, _ in suggestions:
                        # Buat tombol untuk setiap saran
                        if st.button(f"{suggestion} (Confidence: {score}%)"):
                            # Tampilkan informasi berdasarkan jenis query
                            display_information(suggestion, search_term)
