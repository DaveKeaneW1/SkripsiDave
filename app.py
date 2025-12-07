# ============================================
#  IMPORT LIBRARY
# ============================================
import streamlit as st                 # Library utama untuk membuat aplikasi web interaktif berbasis Python
import pandas as pd                   # membaca file CSV/XLSX, membuat DataFrame
import numpy as np                    # Untuk operasi numerik
import plotly.express as px           # Untuk membuat visualisasi interaktif bar dll
from sklearn.preprocessing import MinMaxScaler   # Untuk normalisasi data numerik ke rentang [0, 1]
from sklearn.decomposition import PCA             # Untuk reduksi dimensi (menampilkan data 2D/3D)
from sklearn.cluster import KMeans, DBSCAN        # Dua algoritma clustering utama
from sklearn.metrics import silhouette_score, davies_bouldin_score  # Untuk evaluasi kualitas clustering
import warnings
warnings.filterwarnings('ignore')     # Menonaktifkan warning 


#  KONFIGURASI HALAMAN STREAMLIT
st.set_page_config(
    page_title="FilmCluster - Analysis",   # Judul tab browser
    page_icon="🎬",                        # Icon tab
    layout="wide",                         # Layout lebar penuh
    initial_sidebar_state="expanded",      # Sidebar 
    menu_items={'About': "Film Clustering Analysis"}
)


#  CUSTOM STYLING (CSS untuk sidebar)
st.markdown("""
<style>
    /* Warna gradasi untuk background sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #a855f7 0%, #d946ef 100%);
    }
    
    /* Warna teks sidebar jadi putih */
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Heading sidebar juga putih */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: white !important;
    }
    
    /* Styling tombol radio di sidebar */
    [data-testid="stSidebar"] [role="radio"] {
        color: white !important;
    }
    
    /* Garis pembatas sidebar */
    [data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.2) !important;
    }
</style>
""", unsafe_allow_html=True)


#  INISIALISASI SESSION STATE
# Agar data tidak hilang saat halaman diubah
if 'data' not in st.session_state:
    st.session_state.data = None              # Menyimpan dataset yang diupload
if 'scaled_data' not in st.session_state:
    st.session_state.scaled_data = None       # Menyimpan data hasil normalisasi
if 'clusters' not in st.session_state:
    st.session_state.clusters = None          # Menyimpan label hasil clustering
if 'algorithm' not in st.session_state:
    st.session_state.algorithm = None         # Menyimpan algoritma yang digunakan
if 'pca_model' not in st.session_state:
    st.session_state.pca_model = None         # Menyimpan model PCA untuk visualisasi
if 'selected_features' not in st.session_state:
    st.session_state.selected_features = None # Menyimpan fitur yang dipilih untuk clustering

#  SIDEBAR NAVIGATION
with st.sidebar:
    # Header di sidebar (judul aplikasi)
    st.markdown("""
    <div style='text-align: center; padding: 1.5rem 0; margin-bottom: 1.5rem; border-bottom: 1px solid #dee2e6;'>
        <h2 style='margin: 0; font-size: 2rem; background: linear-gradient(135deg, #000000); 
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;'>
            🎬 FilmCluster
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigasi antar halaman
    page = st.radio(
        "Navigasi",
        ["🏠 Beranda", "📤 Unggah Data", "⚙️ Clustering", "📊 Hasil", "📈 Evaluasi", "💡 Interpretasi"],
        label_visibility="collapsed"
    )
    
    # Footer di sidebar
    st.markdown("---")
    st.markdown("""
    <div style='font-size: 0.75rem; color: #64748b; text-align: center; padding: 1rem 0;'>
        <p>Film Clustering Analysis</p>
    </div>
    """, unsafe_allow_html=True)

#  HALAMAN HOME
if page == "🏠 Beranda":
    # Header dan deskripsi aplikasi
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:12px; padding: 8px 0;">
            <div style="font-size:52px; line-height:1;">🎬</div>
            <h1 style="margin:0; font-size:32px;">Analisis Clustering Film</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Penjelasan singkat fungsi aplikasi
    st.markdown(
    """
    <p style="font-size:20px; line-height:1.6; text-align:justify;">
    Aplikasi ini dirancang untuk melakukan analisis dan pengelompokan film menggunakan <b>K-Means</b> dan <b>DBSCAN</b>. 
    Melalui pengolahan data seperti <i>budget</i>, <i>revenue</i>, <i>ROI</i>, <i>popularity</i>, dan <i>vote average</i>, 
    aplikasi ini membantu pengguna mengidentifikasi pola profitabilitas dan karakteristik setiap film.
    </p>
    <hr>
    """,
    unsafe_allow_html=True
    )

    # Informasi fitur utama
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Format", "CSV, XLSX")
    with col2:
        st.metric("🤖 Algoritma", "K-Means, DBSCAN")
    with col3:
        st.metric("📈 Ukuran Max", "50 MB")
    
    st.markdown("---")
    
    # Dua kolom berisi fitur dan langkah penggunaan
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ✨ Fitur Utama")
        features = [
            "📁 Dukungan CSV & XLSX",
            "🔄 K-Means & DBSCAN",
            "📊 Visualisasi 3D",
            "📈 Metrik Kualitas",
            "💡 Insights"
        ]
        for feature in features:
            st.markdown(f"• {feature}")
    
    with col2:
        st.markdown("### 🚀 Mulai Cepat")
        steps = [
            "1. Unggah dataset Anda",
            "2. Pilih fitur",
            "3. Konfigurasi clustering",
            "4. Lihat hasil",
            "5. Ekspor insights"
        ]
        for step in steps:
            st.markdown(f"• {step}")


#  HALAMAN UPLOAD DATA
elif page == "📤 Unggah Data":
    st.markdown("# 📤 Unggah Data Film")
    st.markdown("---")
    
    # Upload file CSV/XLSX
    uploaded_file = st.file_uploader(
        "Pilih file CSV atau XLSX",
        type=["csv", "xlsx"],
        help="Unggah dataset film Anda"
    )
    
    # Jika file sudah diupload
    if uploaded_file is not None:
        try:
            import requests
            files = {'file': (uploaded_file.name, uploaded_file.getbuffer(), uploaded_file.type)}
            backend_url = "http://backend:5000/api/upload"  # Endpoint Flask untuk upload file
            
            with st.spinner("Mengunggah file..."):
                response = requests.post(backend_url, files=files)  # Kirim file ke backend Flask
            
            if response.status_code == 200:
                result = response.json()  # Ambil nama file yang disimpan backend
                
                # Load file lokal agar bisa ditampilkan
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                # Simpan ke session_state
                st.session_state.data = df
                st.session_state.uploaded_filename = result['filename']
                
                st.success(f"✅ File berhasil diunggah!")
                
                # Tampilkan ringkasan dataset
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📊 Baris", f"{len(df):,}")
                with col2:
                    st.metric("📋 Kolom", len(df.columns))
                with col3:
                    st.metric("🔢 Numerik", len(df.select_dtypes(include=[np.number]).columns))
                with col4:
                    st.metric("📝 Teks", len(df.select_dtypes(include=['object']).columns))
                
                # Preview data
                st.markdown("---")
                st.markdown("### Pratinjau Data")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Statistik dan info kolom
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### Statistik")
                    st.dataframe(df.describe(), use_container_width=True)
                with col2:
                    st.markdown("### Informasi Kolom")
                    col_info = pd.DataFrame({
                        'Kolom': df.columns,
                        'Tipe': df.dtypes,
                        'Non-Null': df.count(),
                        'Null': df.isnull().sum()
                    })
                    st.dataframe(col_info, use_container_width=True)
            else:
                # Jika upload gagal
                error_msg = response.json().get('error', 'Unknown error')
                st.error(f"Gagal mengunggah: {error_msg}")
        
        except requests.exceptions.ConnectionError:
            st.error("Tidak dapat terhubung ke backend. Pastikan Flask berjalan di http://backend:5000")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.info("👆 Unggah file CSV atau XLSX untuk memulai")


#  HALAMAN CLUSTERING
elif page == "⚙️ Clustering":
    st.markdown("# ⚙️ Konfigurasi Clustering")
    st.markdown("---")
    
    # Jika belum ada data, minta upload dulu
    if st.session_state.data is None:
        st.warning("⚠️ Silakan unggah data terlebih dahulu")
    else:
        df = st.session_state.data
        # Ambil hanya kolom numerik (karena clustering butuh angka)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Hapus kolom ID 
        numeric_cols = [col for col in numeric_cols if col.lower() != 'id']
        
        if len(numeric_cols) == 0:
            st.error("❌ Tidak ada kolom numerik yang ditemukan")
        else:
            # Pilih fitur yang akan digunakan untuk clustering
            st.markdown("### 🎯 Pilih Fitur")
            selected_features = st.multiselect(
                "Pilih kolom numerik",
                numeric_cols,
                default=numeric_cols[:min(3, len(numeric_cols))]  # default 3 kolom pertama
            )
            
            # Harus minimal 2 fitur untuk clustering
            if selected_features and len(selected_features) >= 2:
                X = df[selected_features].dropna()    # ambil data fitur tanpa nilai kosong
                cleaned_indices = X.index             # simpan index asli untuk sinkronisasi nanti
                
                # ---------------------------------------
                #  TAMPILKAN DATA SEBELUM NORMALISASI
                # ---------------------------------------
                st.markdown("---")
                st.markdown("### 📊 Normalisasi Data")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Sebelum Normalisasi:**")
                    before_stats = pd.DataFrame({
                        'Fitur': selected_features,
                        'Rata-rata': X.mean().values,
                        'Min': X.min().values,
                        'Max': X.max().values
                    })
                    # Format angka agar lebih rapi
                    before_stats_display = before_stats.copy()
                    for c in ['Rata-rata', 'Min', 'Max']:
                        before_stats_display[c] = before_stats_display[c].apply(lambda x: f"{x:.4f}")
                    st.dataframe(before_stats_display, use_container_width=True, hide_index=True)
                
                # ---------------------------------------
                #  NORMALISASI DENGAN MIN-MAX SCALER
                # ---------------------------------------
                scaler = MinMaxScaler()
                X_scaled = scaler.fit_transform(X)
                
                with col2:
                    st.markdown("**Setelah Normalisasi:**")
                    after_stats = pd.DataFrame({
                        'Fitur': selected_features,
                        'Rata-rata': X_scaled.mean(axis=0),
                        'Min': X_scaled.min(axis=0),
                        'Max': X_scaled.max(axis=0)
                    })
                    after_stats_display = after_stats.copy()
                    for c in ['Rata-rata', 'Min', 'Max']:
                        after_stats_display[c] = after_stats_display[c].apply(lambda x: f"{x:.4f}")
                    st.dataframe(after_stats_display, use_container_width=True, hide_index=True) #Menampilkan 4 angka dibelakang koma
                
                # Simpan ke session agar bisa diakses di halaman lain
                st.session_state.scaled_data = X_scaled
                st.session_state.selected_features = selected_features
                st.session_state.cleaned_indices = cleaned_indices
                
                # ---------------------------------------
                # PILIH ALGORITMA
                # ---------------------------------------
                st.markdown("---")
                st.markdown("### 🔧 Pilih Algoritma")
                
                algorithm = st.radio(
                    "Pilih algoritma",
                    ["K-Means", "DBSCAN"],
                    horizontal=True,
                    label_visibility="collapsed"
                )
                
                st.markdown("---")
                
                # =======================================
                #  K-MEANS CLUSTERING
                # =======================================
                if algorithm == "K-Means":
                    st.markdown("### ⚙️ Parameter K-Means")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        n_clusters = st.slider("Jumlah cluster", 2, 10, 3)
                    with col2:
                        random_state = st.number_input("Random state", value=42)
                    
                    # Jalankan clustering
                    if st.button("🚀 Jalankan K-Means", use_container_width=True, type="primary"):
                        with st.spinner("Menjalankan K-Means..."):
                            kmeans = KMeans(n_clusters=n_clusters, random_state=int(random_state), n_init=10)
                            clusters = kmeans.fit_predict(X_scaled)     # hasil label cluster
                            st.session_state.clusters = clusters
                            st.session_state.kmeans_model = kmeans
                            st.session_state.algorithm = "K-Means"
                            
                            # PCA untuk visualisasi 2D
                            pca = PCA(n_components=2)
                            X_pca = pca.fit_transform(X_scaled)
                            st.session_state.pca_model = pca
                            st.session_state.X_pca = X_pca
                            
                            # Simpan hasil cluster ke backend
                            try:
                                import requests
                                backend_url = "http://backend:5000/api/save-clusters"
                                response = requests.post(backend_url, json={
                                    'filename': st.session_state.uploaded_filename,
                                    'clusters': clusters.tolist(),
                                    'cleaned_indices': st.session_state.cleaned_indices.tolist()
                                })
                            except:
                                pass  # lanjut walau gagal simpan
                            
                            st.success(f"✅ K-Means selesai! Ditemukan {n_clusters} cluster")
                
                # =======================================
                #  DBSCAN CLUSTERING
                # =======================================
                else:
                    st.markdown("### ⚙️ Parameter DBSCAN")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        min_samples = st.slider("Minimum samples", 2,30, 5)
                    with col2:
                        eps = st.slider("Epsilon (eps)", 0.01, 1.5, 0.5, 0.01)
                    
                    if st.button("🚀 Jalankan DBSCAN", use_container_width=True, type="primary"):
                        with st.spinner("Menjalankan DBSCAN..."):
                            dbscan = DBSCAN(eps=eps, min_samples=int(min_samples))
                            clusters = dbscan.fit_predict(X_scaled)
                            st.session_state.clusters = clusters
                            st.session_state.dbscan_model = dbscan
                            st.session_state.algorithm = "DBSCAN"
                            
                            # Reduksi dimensi (untuk visualisasi)
                            pca = PCA(n_components=2)
                            X_pca = pca.fit_transform(X_scaled)
                            st.session_state.pca_model = pca
                            st.session_state.X_pca = X_pca
                            
                            # Simpan hasil ke backend
                            try:
                                import requests
                                backend_url = "http://backend:5000/api/save-clusters"
                                response = requests.post(backend_url, json={
                                    'filename': st.session_state.uploaded_filename,
                                    'clusters': clusters.tolist(),
                                    'cleaned_indices': st.session_state.cleaned_indices.tolist()
                                })
                            except:
                                pass
                            
                            # Hitung jumlah cluster dan noise
                            n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
                            n_noise = list(clusters).count(-1)
                            
                            st.success(f"✅ DBSCAN selesai! {n_clusters} cluster, {n_noise} titik noise")
            else:
                st.warning("⚠️ Pilih minimal 2 fitur")


# HALAMAN HASIL (RESULTS)
elif page == "📊 Hasil":
    st.markdown("# 📊 Hasil Clustering")
    st.markdown("---")
    
    if st.session_state.clusters is None:
        st.warning("⚠️ Jalankan clustering terlebih dahulu")
    else:
        df = st.session_state.data
        clusters = st.session_state.clusters
        cleaned_indices = st.session_state.cleaned_indices
        
        # Tambahkan label cluster ke dataset asli
        full_clusters = np.full(len(df), np.nan)
        full_clusters[cleaned_indices] = clusters
        
        df_with_clusters = df.copy()
        df_with_clusters['Cluster'] = full_clusters
        df_clustered = df_with_clusters[df_with_clusters['Cluster'].notna()].copy()
        df_clustered['Cluster'] = df_clustered['Cluster'].astype(int)
        
        # ---------------------------------------
        # VISUALISASI PCA (2D)
        # ---------------------------------------
        if hasattr(st.session_state, 'X_pca'):
            st.markdown("### 📈 Visualisasi PCA")
            
            pca_df = pd.DataFrame({
                'PC1': st.session_state.X_pca[:, 0],
                'PC2': st.session_state.X_pca[:, 1],
                'Cluster': clusters
            })
            
            fig = px.scatter(
                pca_df, x='PC1', y='PC2', color='Cluster',
                title=f"Hasil Clustering {st.session_state.algorithm}",
                color_continuous_scale="Viridis"
            )
            
            unique_clusters = sorted(pca_df['Cluster'].unique())
            fig.update_layout(
                height=500, 
                template="plotly_dark",
                coloraxis_colorbar=dict(
                    tickvals=unique_clusters,
                    ticktext=[str(int(c)) for c in unique_clusters]
                )
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
        
        # ---------------------------------------
        #  DISTRIBUSI CLUSTER
        # ---------------------------------------
        st.markdown("### 📈 Distribusi Cluster")
        cluster_counts = pd.Series(clusters).value_counts().sort_index()
        
        fig = px.bar(
            x=cluster_counts.index, y=cluster_counts.values,
            labels={'x': 'Cluster', 'y': 'Jumlah'},
            title="Film per Cluster",
            color=cluster_counts.index,
            color_continuous_scale="Viridis"
        )
        fig.update_layout(
            showlegend=False, 
            height=400, 
            template="plotly_dark",
            coloraxis_colorbar=dict(
                tickmode='linear',
                tick0=0,
                dtick=1,
                tickvals=sorted(cluster_counts.index.tolist()),
                ticktext=[str(int(i)) for i in sorted(cluster_counts.index.tolist())]
            )
        )
        fig.update_xaxes(
            tickmode='linear',
            tick0=min(cluster_counts.index),
            dtick=1,
            tickvals=sorted(cluster_counts.index.tolist())
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistik dan export data
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Statistik Cluster")
            cluster_stats = df_clustered.groupby('Cluster').size().reset_index(name='Jumlah')
            cluster_stats['Persentase'] = (cluster_stats['Jumlah'] / len(df_clustered) * 100).round(1)
            st.dataframe(cluster_stats, use_container_width=True)
        
        with col2:
            st.markdown("### 📥 Ekspor Data")
            csv = df_clustered.to_csv(index=False)
            st.download_button(
                label="📥 Unduh CSV",
                data=csv,
                file_name="film_clustering.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        # Preview data lengkap
        st.markdown("---")
        st.markdown("### 👀 Dataset Lengkap")
        st.dataframe(df_clustered, use_container_width=True)


# HALAMAN EVALUATION
elif page == "📈 Evaluasi":
    st.markdown("# 📈 Metrik Evaluasi")
    st.markdown("---")
    
    # Pastikan clustering sudah dijalankan
    if st.session_state.clusters is None:
        st.warning("⚠️ Jalankan clustering terlebih dahulu")
    else:
        # Ambil data hasil scaling dan label cluster dari session
        X_scaled = st.session_state.scaled_data
        clusters = st.session_state.clusters
        
        # ===============================
        # 🔹 Evaluasi untuk DBSCAN
        # ===============================
        if st.session_state.algorithm == "DBSCAN":
            # DBSCAN punya "noise" (label -1) yang harus dihapus sebelum evaluasi
            noise_mask = clusters != -1
            X_eval = X_scaled[noise_mask]
            clusters_eval = clusters[noise_mask]
            
            # Jika jumlah cluster < 2, evaluasi tidak valid
            if len(set(clusters_eval)) < 2:
                st.warning("⚠️ Cluster tidak cukup (tidak termasuk noise) untuk evaluasi")
            else:
                # Hitung metrik evaluasi
                silhouette = silhouette_score(X_eval, clusters_eval)
                davies_bouldin = davies_bouldin_score(X_eval, clusters_eval)
                
                # Tampilkan metrik dalam bentuk "kartu" (st.metric)
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("🎯 Silhouette Score", f"{silhouette:.4f}", help="Semakin tinggi semakin baik (>0.5)")
                with col2:
                    st.metric("📊 Davies-Bouldin Index", f"{davies_bouldin:.4f}", help="Semakin rendah semakin baik (<1.0)")
                with col3:
                    n_noise = list(clusters).count(-1)
                    st.metric("🔴 Titik Noise", n_noise)
        
        # ===============================
        #  Evaluasi untuk K-MEANS
        # ===============================
        else:
            silhouette = silhouette_score(X_scaled, clusters)
            davies_bouldin = davies_bouldin_score(X_scaled, clusters)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("🎯 Silhouette Score", f"{silhouette:.4f}", help="Semakin tinggi semakin baik (>0.5)")
            with col2:
                st.metric("📊 Davies-Bouldin Index", f"{davies_bouldin:.4f}", help="Semakin rendah semakin baik (<1.0)")
        
        # -------------------------------------------
        # Penjelasan interpretasi metrik secara visual
        # -------------------------------------------
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🎯 Silhouette Score
            - **> 0.5**: ✅ Bagus → cluster terbentuk jelas dan terpisah.
            - **0.25 - 0.5**: ⚠️ Cukup → cluster agak tumpang tindih.
            - **< 0.25**: ❌ Buruk → cluster buruk atau data acak.
            """)
        
        with col2:
            st.markdown("""
            ### 📊 Davies-Bouldin Index
            - **< 1.0**: ✅ Sempurna → cluster sangat terpisah.
            - **1.0 - 2.0**: ✅ Bagus → cukup bagus.
            - **> 2.0**: ❌ Buruk → cluster tumpang tindih atau tidak jelas.
            """)


# ============================================
# HALAMAN INSIGHTS (INTERPRETASI HASIL)
# ============================================
elif page == "💡 Interpretasi":
    st.markdown("# 💡 Interpretasi Hasil")
    st.markdown("Insight dan karakteristik dari hasil clustering")
    st.markdown("---")
    
    if st.session_state.clusters is None:
        st.warning("⚠️ Jalankan clustering terlebih dahulu")
    else:
        df = st.session_state.data
        clusters = st.session_state.clusters
        cleaned_indices = st.session_state.cleaned_indices
        
        # Tambahkan label cluster ke dataset asli
        df_with_clusters = df.copy()
        full_clusters = np.full(len(df), np.nan)
        full_clusters[cleaned_indices] = clusters
        df_with_clusters['Cluster'] = full_clusters
        
        # Ambil hanya kolom numerik untuk analisis statistik
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Warna untuk tiap cluster (agar konsisten di UI)
        colors = {
            0: "#3b82f6",  # Blue
            1: "#10b981",  # Green
            2: "#f59e0b",  # Orange
            3: "#8b5cf6",  # Purple
            4: "#ec4899",  # Pink
            5: "#06b6d4",  # Cyan
            6: "#f97316",  # Orange-red
            7: "#6366f1",  # Indigo
            -1: "#6b7280"  # Gray for noise
        }
        
        try:
            import requests
            # Endpoint backend Flask untuk analisis insight otomatis
            backend_url = "http://backend:5000/api/cluster-insights"
            response = requests.post(backend_url, json={
                'filename': st.session_state.uploaded_filename
            })
            
            # Jika permintaan berhasil (status 200)
            if response.status_code == 200:
                insights_data = response.json()['insights']
                
                st.markdown("### 🎯 Karakteristik Sifat Cluster")
                
                # Loop setiap cluster dan tampilkan hasil analisis
                for idx, insight in enumerate(insights_data):
                    cluster_id = insight['cluster_id']
                    size = insight['size']
                    pct = insight['percentage']
                    summary = insight['summary']
                    characteristics = insight['characteristics']
                    stats = insight['statistics']
                    genre_analysis = insight['genre_analysis']
                    
                    color = colors.get(cluster_id, "#3b82f6")
                    
                    # -----------------------------------------------
                    #  Tampilan kartu ringkasan tiap cluster
                    # -----------------------------------------------
                    st.markdown(f"""
                    <div style='
                        background: linear-gradient(135deg, {color}15 0%, {color}05 100%);
                        border-left: 4px solid {color};
                        border-radius: 8px;
                        padding: 1.5rem;
                        margin-bottom: 1.5rem;
                        border: 1px solid {color}30;
                    '>
                        <div style='display: flex; justify-content: space-between; align-items: start; margin-bottom: 1rem;'>
                            <div>
                                <h3 style='margin: 0; color: {color}; font-size: 1.2rem;'>{summary}</h3>
                                <p style='margin: 0.5rem 0 0 0; color: #94a3b8; font-size: 0.9rem;'>{size} film • {pct:.1f}% dari total</p>
                            </div>
                            <div style='
                                background: {color}20;
                                color: {color};
                                padding: 0.5rem 1rem;
                                border-radius: 6px;
                                font-weight: 600;
                                font-size: 0.9rem;
                            '>
                                Cluster {cluster_id if cluster_id != -1 else "Noise"}
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # -----------------------------------------------
                    #  Kolom 1: karakteristik utama tiap cluster
                    #  Kolom 2: genre dominan
                    # -----------------------------------------------
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**📊 Karakteristik Utama:**")
                        if characteristics:
                            for char in characteristics:
                                st.markdown(f"• {char}")
                        else:
                            st.markdown("• Tidak ada karakteristik khusus")
                    
                    with col2:
                        st.markdown("**🎬 Genre Dominan:**")
                        if genre_analysis and genre_analysis.get('dominant_genres'):
                            genres = genre_analysis['dominant_genres']
                            if isinstance(genres, list) and len(genres) > 0:
                                st.markdown(f"• {genres[0]}")
                            elif isinstance(genres, str):
                                st.markdown(f"• {genres}")
                            else:
                                st.markdown("• Data genre tidak tersedia")
                        else:
                            st.markdown("• Data genre tidak tersedia")
                    
                    # -----------------------------------------------
                    #  Statistik detail (min, max, mean, std)
                    # -----------------------------------------------
                    with st.expander("📈 Statistik Detail", expanded=False):
                        stats_df = pd.DataFrame([
                            {
                                'Fitur': feature,
                                'Rata-rata': f"{data['mean']:.2f}",
                                'Min': f"{data['min']:.2f}",
                                'Max': f"{data['max']:.2f}",
                                'Std': f"{data['std']:.2f}"
                            }
                            for feature, data in stats.items()
                        ])
                        st.dataframe(stats_df, use_container_width=True)
                        
                        # Distribusi genre (opsional)
                        if genre_analysis and genre_analysis.get('genre_distribution'):
                            st.markdown("**Distribusi Genre:**")
                            genre_df = pd.DataFrame([
                                {'Genre': g, 'Jumlah': c}
                                for g, c in sorted(genre_analysis['genre_distribution'].items(), key=lambda x: x[1], reverse=True)
                            ])
                            st.dataframe(genre_df, use_container_width=True)
                    
                    st.markdown("")
                
                # -----------------------------------------------
                #  RINGKASAN KESELURUHAN
                # -----------------------------------------------
                st.markdown("---")
                st.markdown("### 📋 Ringkasan Keseluruhan")
                
                summary_col1, summary_col2, summary_col3 = st.columns(3)
                
                with summary_col1:
                    st.metric("Total Cluster", len(insights_data))
                with summary_col2:
                    total_films = sum(i['size'] for i in insights_data)
                    st.metric("Total Film", total_films)
                with summary_col3:
                    noise_count = sum(i['size'] for i in insights_data if i['cluster_id'] == -1)
                    st.metric("Titik Noise", noise_count if noise_count > 0 else 0)
            
            else:
                # Jika backend gagal mengirim respon
                st.error("Tidak dapat mengambil insights dari backend")
        
        except Exception as e:
            # Jika ada error (misal backend mati)
            st.error(f"Error mengambil insights: {str(e)}")
