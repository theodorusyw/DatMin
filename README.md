# Bank Customer Insight Dashboard — Phase 5

Dashboard interaktif (Python Dash) yang memenuhi requirement **Phase 5: Visualization and Knowledge Presentation** dari project data mining Bank Customer Churn.

## Isi Dashboard (5 Tab)

1. **Ringkasan Eksekutif** — KPI utama + jawaban langsung atas pertanyaan sentral phase ini.
2. **Segmentasi Pelanggan** — Peta klaster (PCA), pilih metode (K-Modes 3 segmen, K-Modes 2, K-Means 2, DBSCAN), profil tiap segmen, distribusi fitur per segmen.
3. **Pattern Mining** — Jaringan aturan asosiasi (rule network) interaktif, slider untuk mengatur min support/confidence, tabel lengkap aturan (Apriori).
4. **Deteksi Anomali** — Peta anomali (Isolation Forest) di ruang PCA, perbandingan anomali vs klaster, distribusi fitur Normal vs Anomaly.
5. **Knowledge Discovery Report** — Laporan temuan dalam bahasa bisnis (non-teknis), termasuk rekomendasi tindakan.

Semua angka pada dashboard dihitung otomatis dari `data/after_phase4.csv`, bukan angka statis — jika kamu meng-generate ulang CSV dari notebook, tinggal ganti file di folder `data/` dan jalankan ulang.

## Cara Menjalankan

1. Pastikan Python 3.9+ terpasang.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Jalankan aplikasi:
   ```bash
   python app.py
   ```
4. Buka browser ke `http://127.0.0.1:8050`

> Catatan: Jika `mlxtend` belum ter-install, aplikasi tetap bisa jalan — ia otomatis memakai mesin pencarian pola pairwise-apriori sederhana sebagai cadangan. Tapi untuk hasil yang identik dengan notebook Phase 3, disarankan tetap install `mlxtend`.

## Struktur Folder

```
phase5_dashboard/
├── app.py                # Aplikasi Dash utama
├── requirements.txt
├── README.md
├── assets/
│   └── custom.css        # Styling dashboard
└── data/
    └── after_phase4.csv  # Data gabungan hasil Phase 1-4
```

## Sumber Data

`after_phase4.csv` adalah gabungan hasil dari seluruh phase sebelumnya:
- Fitur dasar & binning (Phase 1)
- Kolom `cluster`, `cluster_kmodes`, `cluster_kmeans`, `dbscan_cluster` (Phase 2)
- Kolom dasar yang sama dipakai ulang untuk pattern mining (Phase 3, dihitung langsung di dashboard)
- Kolom `Anomaly` dari Isolation Forest (Phase 4)

## Kustomisasi Lanjutan

- Ambang batas pattern mining (support/confidence) bisa diubah langsung dari slider di tab Pattern Mining — tidak perlu edit kode.
- Palet warna & font ada di `assets/custom.css`.
- Jika ingin menambahkan tab baru, tambahkan fungsi `render_tab_xxx()` baru dan daftarkan di `dcc.Tabs` + callback `render_tab`.

---
Butuh versi laporan **Knowledge Discovery Report** sebagai dokumen Word terpisah untuk presentasi 10 menit? Tinggal minta ke Claude.
