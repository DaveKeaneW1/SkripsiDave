from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
from werkzeug.utils import secure_filename


# Inisialisasi Flask
app = Flask(__name__)
CORS(app)  # izinkan akses dari frontend (Streamlit)

# Folder penyimpanan file upload
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}  # format file yang diterima

# Buat folder uploads jika belum ada
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Konfigurasi aplikasi
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # Maksimal 50 MB


#  Fungsi bantu: cek ekstensi file
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


#  Fungsi bantu: baca file CSV/XLSX ke pandas DataFrame
def load_data(filepath):
    """Membaca dataset berdasarkan format file."""
    if filepath.endswith('.xlsx'):
        df = pd.read_excel(filepath)
    else:
        df = pd.read_csv(filepath)
    return df

#  Endpoint: Upload dataset
@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Upload file CSV/XLSX dan validasi isinya."""
    try:
        # Pastikan ada file di request
        if 'file' not in request.files:
            return jsonify({'error': 'Tidak ada file'}), 400

        file = request.files['file']

        # Pastikan file dipilih
        if file.filename == '':
            return jsonify({'error': 'File tidak dipilih'}), 400

        # Pastikan format sesuai
        if not allowed_file(file.filename):
            return jsonify({'error': 'Format file harus CSV atau XLSX'}), 400

        # Amankan nama file sebelum disimpan
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Baca isi dataset
        df = load_data(filepath)

        # Kirim metadata ke frontend
        return jsonify({
            'success': True,
            'filename': filename,
            'rows': len(df),
            'columns': list(df.columns),
            'data_preview': df.head(5).to_dict('records')  # tampilan 5 baris awal
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

#  Endpoint: Analisis insight otomatis per cluster
@app.route('/api/cluster-insights', methods=['POST'])
def cluster_insights():
    """Menganalisis karakteristik tiap cluster (high/low/medium)."""
    try:
        data = request.json
        filename = data.get('filename')

        if not filename:
            return jsonify({'error': 'Filename tidak ditemukan'}), 400

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'File tidak ditemukan'}), 404

        # Baca data
        df = load_data(filepath)

        # Pastikan cluster sudah ada
        if 'cluster' not in df.columns:
            return jsonify({'error': 'Cluster column tidak ditemukan. Jalankan clustering terlebih dahulu'}), 400

        # Siapkan data cluster
        df_clustered = df[df['cluster'].notna()].copy()
        df_clustered['cluster'] = df_clustered['cluster'].astype(int)

        if len(df_clustered) == 0:
            return jsonify({'error': 'Tidak ada data cluster yang valid'}), 400

        insights = []
        # Ambil kolom numerik (selain cluster & id)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in ['cluster', 'id']]

        # Loop tiap cluster
        for cluster_id in sorted(set(df_clustered['cluster'])):
            cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
            cluster_size = len(cluster_data)
            cluster_pct = (cluster_size / len(df_clustered) * 100)

            stats = {}
            characteristics = []

            # Hitung statistik tiap kolom numerik
            for col in numeric_cols:
                if col in cluster_data.columns:
                    mean_val = float(cluster_data[col].mean())
                    overall_mean = float(df_clustered[col].mean())
                    overall_std = float(df_clustered[col].std())

                    stats[col] = {
                        'mean': mean_val,
                        'min': float(cluster_data[col].min()),
                        'max': float(cluster_data[col].max()),
                        'std': float(cluster_data[col].std())
                    }

                    # Menggunakan z-score dengan threshold yang lebih jelas dan tidak overlap
                    if overall_std > 0:
                        z_score = (mean_val - overall_mean) / overall_std
                        
                        # Definisi yang lebih jelas:
                        # HIGH: z_score > 0.5 (jauh di atas rata-rata)
                        # MEDIUM: -0.5 <= z_score <= 0.5 (sekitar rata-rata)
                        # LOW: z_score < -0.5 (jauh di bawah rata-rata)
                        if z_score > 0.5:
                            characteristics.append(f"High {col}")
                        elif z_score < -0.5:
                            characteristics.append(f"Low {col}")
                        else:
                            characteristics.append(f"Medium {col}")
                    else:
                        # Jika std = 0 (semua data sama), kategorikan sebagai Medium
                        characteristics.append(f"Medium {col}")

            # Analisis genre
            genre_analysis = {}
            if 'genres' in cluster_data.columns:
                all_genres = []
                for genres_str in cluster_data['genres'].dropna():
                    if isinstance(genres_str, str):
                        genres = [g.strip() for g in str(genres_str).split(',')]
                        all_genres.extend(genres)
                if all_genres:
                    from collections import Counter
                    genre_counts = Counter(all_genres)
                    sorted_genres = sorted(genre_counts.items(), key=lambda x: (-x[1], x[0]))
                    dominant_genre = [sorted_genres[0][0]] if sorted_genres else []
                    genre_analysis = {
                        'dominant_genres': dominant_genre,
                        'genre_distribution': dict(sorted_genres[:5])
                    }

            # Ringkasan tiap cluster
            summary = f"Cluster {cluster_id}: {cluster_size} films ({cluster_pct:.1f}%)"
            if characteristics:
                summary += f" - {', '.join(characteristics[:3])}"

            insights.append({
                'cluster_id': int(cluster_id),
                'size': cluster_size,
                'percentage': float(cluster_pct),
                'summary': summary,
                'characteristics': characteristics,
                'statistics': stats,
                'genre_analysis': genre_analysis
            })

        return jsonify({'success': True, 'insights': insights}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

#  Endpoint: Simpan hasil clustering ke file asli
@app.route('/api/save-clusters', methods=['POST'])
def save_clusters():
    """Menyimpan label cluster ke dataset asli (sinkron dengan indeks)."""
    try:
        data = request.json
        filename = data.get('filename')
        clusters = data.get('clusters')
        cleaned_indices = data.get('cleaned_indices')

        if not filename or clusters is None:
            return jsonify({'error': 'Missing filename or clusters'}), 400

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'File tidak ditemukan'}), 404

        # Baca data asli
        df = load_data(filepath)

        # Buat kolom cluster dengan NaN default
        full_clusters = np.full(len(df), np.nan)
        full_clusters[cleaned_indices] = clusters  # isi cluster hanya di baris valid
        df['cluster'] = full_clusters

        # Simpan kembali
        if filename.endswith('.xlsx'):
            df.to_excel(filepath, index=False)
        else:
            df.to_csv(filepath, index=False)

        return jsonify({'success': True, 'message': 'Clusters saved successfully'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

#  Endpoint: Health check (cek apakah server hidup)
@app.route('/api/health', methods=['GET'])
def health():
    """Menjawab status server untuk monitoring."""
    response = jsonify({'status': 'ok'})
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response, 200


#  Jalankan Flask app
if __name__ == '__main__':
    debug_mode = os.getenv('FLASK_DEBUG', 'False') == 'True'
    # host='0.0.0.0' → agar bisa diakses dari semua IP
    app.run(debug=debug_mode, host='0.0.0.0', port=5000, threaded=True)
