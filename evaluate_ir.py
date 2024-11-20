# evaluate_ir.py

import numpy as np
import pandas as pd
from py2neo import Graph
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from rapidfuzz import process, fuzz
import re

# Inisialisasi tokenizer dan model untuk NER
tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
model = AutoModelForTokenClassification.from_pretrained("indobenchmark/indobert-base-p1")
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Fungsi untuk normalisasi nama penyakit
def normalize_disease_name(disease):
    disease = disease.lower().strip()
    disease = re.sub(r'\bsakit\s+', '', disease)
    disease = re.sub(r'\bpenyakit\s+', '', disease)
    return disease

# Fungsi untuk menyelesaikan entitas (entity resolution)
def resolve_entity(name):
    disease_mapping = {
        "migrain": "migraine",
        "kepala pusing": "sakit kepala",
        "sakit kepala": "sakit kepala",
        # Tambahkan mapping lain sesuai kebutuhan
    }
    return disease_mapping.get(name, name)

# Fungsi untuk mengekstrak kelompok usia
def extract_age_group(text):
    text = text.lower()
    if any(word in text for word in ['anak', 'bayi', 'pediatrik']):
        return 'anak-anak'
    elif any(word in text for word in ['dewasa', 'adult']):
        return 'dewasa'
    return 'umum'

# Fungsi untuk mengekstrak penyakit dengan konteks menggunakan Transformers
common_diseases = [
    'flu', 'demam', 'pusing', 'sakit kepala', 'batuk', 'pilek',
    'asma', 'diare', 'tipes', 'maag', 'migraine', 'migrain',
    'hipertensi', 'diabetes', 'kolesterol tinggi', 'ginjal batu',
    'anemia', 'hiv', 'hepatitis', 'alergi', 'infeksi saluran kemih',
    # Tambahkan penyakit lain sesuai kebutuhan
]

additional_disease_keywords = [
    'hipertensi', 'diabetes', 'kolesterol tinggi', 'ginjal batu',
    'anemia', 'hiv', 'hepatitis', 'alergi', 'infeksi saluran kemih',
    'stroke', 'kanker', 'asthma', 'menular', 'tidak stabil',
    # Tambahkan kata kunci lainnya
]

def extract_diseases_from_uses(text):
    if pd.isna(text):
        return []
    
    text = str(text).lower()
    diseases = []
    
    # Contoh pola yang bisa ditangkap, sesuaikan dengan data Anda
    disease_patterns = [
        r'mengobati ([\w\s]+)', 
        r'untuk ([\w\s]+)', 
        r'digunakan untuk ([\w\s]+)'
    ]
    
    for pattern in disease_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            disease = resolve_entity(normalize_disease_name(match.strip()))
            diseases.append({
                'name': disease,
                'age_group': extract_age_group(text)
            })
    
    return diseases

def extract_diseases(text):
    if pd.isna(text):
        return []
    
    text = str(text).lower()
    entities = ner_pipeline(text)
    
    diseases = []
    
    # Ekstraksi dari common_diseases
    for disease in common_diseases:
        if disease in text:
            resolved_name = resolve_entity(normalize_disease_name(disease))
            diseases.append({
                'name': resolved_name,
                'age_group': extract_age_group(text)
            })
    
    # Ekstraksi menggunakan NER
    for ent in entities:
        if ent['entity_group'] in ['DISEASE', 'CONDITION', 'SYMPTOM']:
            normalized_name = normalize_disease_name(ent['word'])
            resolved_name = resolve_entity(normalized_name)
            diseases.append({
                'name': resolved_name,
                'age_group': extract_age_group(text)
            })
    
    # Ekstraksi tambahan menggunakan keyword matching
    for keyword in additional_disease_keywords:
        if keyword in text and keyword not in [d['name'] for d in diseases]:
            resolved_name = resolve_entity(normalize_disease_name(keyword))
            diseases.append({
                'name': resolved_name,
                'age_group': extract_age_group(text)
            })
    
    # Ekstraksi menggunakan regex
    regex_diseases = extract_diseases_from_uses(text)
    for disease_info in regex_diseases:
        if disease_info['name'] not in [d['name'] for d in diseases]:
            diseases.append(disease_info)
    
    # Logging untuk debugging
    print(f"Ekstraksi penyakit dari teks: {text}")
    print(f"Penyakit yang terdeteksi: {[d['name'] for d in diseases]}")
    
    return diseases

# Koneksi ke Neo4j (hanya baca)
graph = Graph("bolt://localhost:7687", auth=("neo4j", "Attaqy81"), secure=False)

# Fungsi untuk mencari obat berdasarkan penyakit
def search_drugs_by_disease(disease):
    query = """
    MATCH (d:Drug)-[:treats]->(dis:Disease {name: $disease})
    RETURN d.name AS drug_name
    """
    result = graph.run(query, disease=disease)
    return [record['drug_name'] for record in result]

# Fungsi untuk mendapatkan semua nama obat untuk fuzzy matching dan auto-complete
def get_all_drug_names():
    query = "MATCH (d:Drug) RETURN d.name AS name"
    result = graph.run(query)
    return [record['name'] for record in result]

# Fungsi untuk mendapatkan atribut spesifik (hanya baca)
def get_drug_price(drug_name):
    query = """
    MATCH (d:Drug {name: $drug_name})
    RETURN d.min_price AS MinPrice, d.max_price AS MaxPrice
    """
    result = graph.run(query, drug_name=drug_name)
    df = pd.DataFrame([record for record in result])
    return df[['MinPrice', 'MaxPrice']].values.tolist()

def get_drug_composition(drug_name):
    query = """
    MATCH (d:Drug {name: $drug_name})-[:contains]->(n:Ingredients)
    RETURN n.content AS Composition
    """
    result = graph.run(query, drug_name=drug_name)
    df = pd.DataFrame([record for record in result])
    return df['Composition'].tolist()

def get_drug_side_effects(drug_name):
    query = """
    MATCH (d:Drug {name: $drug_name})-[:hasSideEffect]->(n:SideEffect)
    RETURN n.effect AS SideEffects
    """
    result = graph.run(query, drug_name=drug_name)
    df = pd.DataFrame([record for record in result])
    return df['SideEffects'].tolist()

def get_drug_manufacturer(drug_name):
    query = """
    MATCH (d:Drug {name: $drug_name})-[:producedBy]->(n:Manufacturer)
    RETURN n.name AS Manufacturer
    """
    result = graph.run(query, drug_name=drug_name)
    df = pd.DataFrame([record for record in result])
    return df['Manufacturer'].tolist()

def get_drug_precautions(drug_name):
    query = """
    MATCH (d:Drug {name: $drug_name})-[:requiresPrecaution]->(n:Precaution)
    RETURN n.details AS Precautions
    """
    result = graph.run(query, drug_name=drug_name)
    df = pd.DataFrame([record for record in result])
    return df['Precautions'].tolist()

# Kelas IRSystem yang meniru struktur metode evaluasi Anda
class IRSystem:
    def boolean_search(self, dataset, query, column):
        # Implementasi pencarian Boolean
        # Contoh sederhana: mencari substring dalam kolom tertentu
        return dataset[column].str.contains(query, case=False).astype(int).tolist()
    
    def vsm_search(self, dataset, query, column):
        # Implementasi Vector Space Model
        # Placeholder: implementasikan sesuai kebutuhan
        # Misalnya, menggunakan TF-IDF dan cosine similarity
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(dataset[column])
        query_vec = vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
        return (similarities > 0.1).astype(int).tolist()  # Threshold bisa disesuaikan
    
    def probabilistic_search(self, dataset, query, column):
        # Implementasi probabilistic search
        # Placeholder: implementasikan sesuai kebutuhan
        return self.vsm_search(dataset, query, column)  # Menggunakan VSM sebagai contoh
    
    def cosine_similarity_search(self, dataset, query, column):
        # Implementasi cosine similarity
        return self.vsm_search(dataset, query, column)  # Menggunakan VSM sebagai contoh
    
    def jaccard_similarity_search(self, dataset, query, column):
        # Implementasi Jaccard Similarity
        # Placeholder: implementasikan sesuai kebutuhan
        def jaccard_similarity(text1, text2):
            a = set(text1.split())
            b = set(text2.split())
            c = a.intersection(b)
            return float(len(c)) / (len(a) + len(b) - len(c))
        
        similarities = dataset[column].apply(lambda x: jaccard_similarity(x, query))
        return (similarities > 0.1).astype(int).tolist()  # Threshold bisa disesuaikan
    
    def evaluate_metrics(self, ground_truth, predictions):
        # Menghitung Precision, Recall, dan F1-Score
        precision = np.sum((ground_truth == 1) & (predictions == 1)) / np.sum(predictions == 1) if np.sum(predictions == 1) > 0 else 0
        recall = np.sum((ground_truth == 1) & (predictions == 1)) / np.sum(ground_truth == 1) if np.sum(ground_truth == 1) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return precision, recall, f1
    
    def get_relevant_stats(self, ground_truth, predictions):
        true_positives = np.sum((ground_truth == 1) & (predictions == 1))
        total_retrieved = np.sum(predictions == 1)
        return true_positives, total_retrieved

# Fungsi untuk memuat ground truth
def load_ground_truth(file_path):
    return pd.read_excel(file_path)

# Fungsi untuk menghitung metrik evaluasi
def evaluate_system(ground_truth_df, system_results):
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    ir_system = IRSystem()
    
    for index, row in ground_truth_df.iterrows():
        query = row['Query']
        expected = set([drug.strip() for drug in row['Expected_Drugs'].split(',')])
        retrieved = set(system_results.get(query, []))
        
        true_positives = len(expected & retrieved)
        precision = true_positives / len(retrieved) if len(retrieved) > 0 else 0
        recall = true_positives / len(expected) if len(expected) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        
        print(f"Query: {query}")
        print(f"Expected: {expected}")
        print(f"Retrieved: {retrieved}")
        print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}\n")
    
    evaluation_metrics = {
        'Precision': precision_scores,
        'Recall': recall_scores,
        'F1-Score': f1_scores
    }
    
    evaluation_df = pd.DataFrame(evaluation_metrics)
    print("Average Precision:", evaluation_df['Precision'].mean())
    print("Average Recall:", evaluation_df['Recall'].mean())
    print("Average F1-Score:", evaluation_df['F1-Score'].mean())
    
    return evaluation_df

# Fungsi untuk mengumpulkan hasil sistem berdasarkan ground truth dan metode IR
def get_system_results(ground_truth_df, ir_system, method_name):
    system_results = {}
    for index, row in ground_truth_df.iterrows():
        query = row['Query']
        # Panggil metode pencarian berdasarkan method_name
        if method_name == 'Boolean':
            predictions = ir_system.boolean_search(dataset, query, column)
        elif method_name == 'Vector Space Model':
            predictions = ir_system.vsm_search(dataset, query, column)
        elif method_name == 'Probabilistic':
            predictions = ir_system.probabilistic_search(dataset, query, column)
        elif method_name == 'Cosine Similarity':
            predictions = ir_system.cosine_similarity_search(dataset, query, column)
        elif method_name == 'Jaccard Similarity':
            predictions = ir_system.jaccard_similarity_search(dataset, query, column)
        else:
            predictions = search_query(query)
        
        # Implementasi top_k filter jika diperlukan
        if sum(predictions) < 4:
            top_k_indices = np.argsort(predictions)[-4:]
            predictions = np.array(predictions)
            predictions[top_k_indices] = 1
            predictions = predictions.tolist()
        
        # Konversi ke daftar obat berdasarkan predictions
        if method_name in ['Boolean', 'Vector Space Model', 'Probabilistic', 'Cosine Similarity', 'Jaccard Similarity']:
            retrieved = [drug for drug, pred in zip(get_all_drug_names(), predictions) if pred == 1]
        else:
            retrieved = predictions  # Untuk metode lain yang mungkin berbeda
        
        system_results[query] = retrieved
    
    return system_results

# Fungsi untuk menjalankan evaluasi pada semua dataset dan metode
def evaluate_all_datasets():
    ir_system = IRSystem()
    results = []

    # Definisikan dataset dan ground truth
    datasets = [
        ('Dataset Osteoarthritis', pd.read_csv('obat_osteoarthritis.csv').fillna(''), 
         'uses', "obat untuk osteoarthritis"),
        ('Dataset Bronkitis', pd.read_csv('obat_bronkitis.csv').fillna(''), 
         'uses', "obat untuk bronkitis"),
        ('Dataset Pertumbuhan Tulang', pd.read_csv('obat_pertumbuhan_tulang.csv').fillna(''), 
         'uses', "obat untuk pertumbuhan tulang"),
        ('Dataset Obat Tidur Anak', pd.read_csv('obat_tidur_anak.csv').fillna(''), 
         'uses', "obat tidur yang aman untuk anak di bawah 5 tahun"),
        ('Dataset Obat Alergi', pd.read_csv('obat_alergi_tidak_bikin_ngantuk.csv').fillna(''), 
         'Indikasi Umum', "obat alergi tanpa efek samping mengantuk untuk dewasa"),
        ('Dataset Obat Flu Ibu Hamil', pd.read_csv('obat_flu_ibu_hamil.csv').fillna(''), 
         'uses', "obat flu dengan dosis rendah yang bisa diminum ibu hamil")
    ]

    # Definisikan metode evaluasi
    methods = [
        ('Boolean', ir_system.boolean_search),
        ('Vector Space Model', ir_system.vsm_search),
        ('Probabilistic', ir_system.probabilistic_search),
        ('Cosine Similarity', ir_system.cosine_similarity_search),
        ('Jaccard Similarity', ir_system.jaccard_similarity_search)
    ]

    # Iterasi melalui setiap dataset
    for dataset_name, dataset, column, query in datasets:
        print(f"Evaluating {dataset_name}")
        # Contoh Ground Truth: sesuaikan dengan ground truth yang relevan
        ground_truth_path = 'ground_truth.xlsx'  # Pastikan path benar
        ground_truth_df = load_ground_truth(ground_truth_path)
        gt_row = ground_truth_df[ground_truth_df['Query'] == query]
        if gt_row.empty:
            print(f"No ground truth found for query: {query}\n")
            continue
        ground_truth = gt_row.iloc[0]['Expected_Drugs'].split(',')
        ground_truth = [drug.strip() for drug in ground_truth]
        ground_truth_count = len(ground_truth)
        print(f"Ground Truth Documents: {ground_truth_count}")

        for method_name, method_func in methods:
            try:
                print(f"Method: {method_name}")
                predictions = method_func(dataset, query, column)
                
                if sum(predictions) < 4:
                    top_k_indices = np.argsort(predictions)[-4:]
                    predictions = np.array(predictions)
                    predictions[top_k_indices] = 1
                    predictions = predictions.tolist()
                
                # Konversi predictions ke daftar obat
                retrieved = [drug for drug, pred in zip(get_all_drug_names(), predictions) if pred == 1]
                
                # Hitung metrik
                true_positives = len(set(retrieved) & set(ground_truth))
                total_retrieved = len(retrieved)
                precision = true_positives / total_retrieved if total_retrieved > 0 else 0
                recall = true_positives / ground_truth_count if ground_truth_count > 0 else 0
                f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                results.append({
                    'Dataset': dataset_name,
                    'Method': method_name,
                    'Precision': precision,
                    'Recall': recall,
                    'F1-Score': f1,
                    'Ground Truth Documents': ground_truth_count,
                    'Retrieved Documents': total_retrieved,
                    'Relevant Retrieved': true_positives
                })
                
            except Exception as e:
                print(f"Error in {method_name}: {str(e)}\n")

    results_df = pd.DataFrame(results)
    return results_df

# Fungsi untuk menyimpan hasil evaluasi
def save_evaluation_results(evaluation_df, output_path='evaluation_results.xlsx'):
    evaluation_df.to_excel(output_path, index=False)
    print(f"Evaluasi disimpan di {output_path}")

# Fungsi utama untuk menjalankan evaluasi
def main():
    results_df = evaluate_all_datasets()
    
    print("\nHasil Evaluasi:")
    print(results_df.to_string(index=False))
    
    # Simpan hasil evaluasi ke CSV
    results_df.to_csv('hasil_evaluasi_information_retrieval.csv', index=False)
    
    # Analisis performa metode
    method_analysis = results_df.groupby('Method')[['Precision', 'Recall', 'F1-Score']].mean()
    best_method = method_analysis['F1-Score'].idxmax()
    
    print("\nMethod Performance Analysis:")
    print(method_analysis)
    print(f"\nBest performing method overall: {best_method}")
    
    # Simpan hasil analisis ke Excel
    method_analysis.to_excel('method_performance_analysis.xlsx')

if __name__ == "__main__":
    main()
