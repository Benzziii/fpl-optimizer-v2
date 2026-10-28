# FPL Quantum Optimizer & Multi-Horizon Predictor (Full-Stack Package)

Paket ini berisi kode sumber lengkap backend Python (`FastAPI` + `PuLP` MILP Solver + `FPL Official API Client`) yang mendampingi aplikasi web interaktif **FPL Quantum Optimizer (`fpl_optimizer_pro.html`)**.

## Struktur File
1. `fpl_client.py` — Pengambil data langsung dari endpoint resmi FPL (`bootstrap-static`, `fixtures`, `element-summary`, dan `entry/{manager_id}`).
2. `xp_predictor.py` — Mesin kalkulasi Expected Points (`xP`) multi-variabel:
   - Harga pemain & nilai efisiensi (`xP / £m`)
   - Status kebugaran pemain (`Fit`, `Doubtful`, `Injured`, `Suspended`) & persentase *Chance to Play*
   - *Set-Piece Takers* (Urutan eksekutor Penalti, *Direct Free Kick*, dan *Corner Kick*)
   - Risiko akumulasi kartu kuning (4 YC menjelang batas GW19 = risiko larangan 1 laga; 8–9 YC menjelang GW32 = risiko larangan 2 laga)
   - Rekor *Head-to-Head* (H2H) historis & keunggulan bermain Kandang/Tandang (*Home/Away*)
   - Statistik bergulir 10–20 pertandingan terakhir untuk pemain (`xG/90`, `xA/90`, `BPS/90`, `xMin`) dan tim (`xGF`, `xGA`)
3. `optimizer_milp.py` — Solver *Mixed-Integer Linear Programming* (MILP) menggunakan `PuLP` (CBC) untuk:
   - Optimasi horizon pendek (**1 GW**), menengah (**5 GW**), hingga panjang (**12 GW**)
   - Perbandingan otomatis antara **Roll Transfer (0 Transfer)**, **Gunakan Free Transfer**, dan **Multi-Transfer dengan Hit (-4/-8 poin)**
   - Alokasi optimal 4 Chip (**Triple Captain**, **Wildcard**, **Free Hit**, **Bench Boost**) pada Paruh Musim 1 (GW1–19) dan Paruh Musim 2 (GW20–38).
4. `app.py` — Server REST API berbasis `FastAPI` dengan dukungan CORS penuh.

## Cara Menjalankan Secara Lokal (Step-by-Step)

### Langkah 1: Persiapkan Lingkungan Python
Pastikan Anda memiliki Python 3.10+ terinstal, lalu buat virtual environment dan instal seluruh dependensi:
```bash
python3 -m venv venv
source venv/bin/activate  # Untuk Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Langkah 2: Jalankan Server Backend & MILP Solver
```bash
uvicorn app:app --reload --port 8000
```
Server akan berjalan di `http://localhost:8000` (dokumentasi Swagger interaktif tersedia di `http://localhost:8000/docs`).

### Langkah 3: Buka Aplikasi Web (`fpl_optimizer_pro.html`)
Anda dapat langsung menggunakan aplikasi web interaktif di browser. Masukkan **FPL Manager ID** Anda (angka yang tertera pada URL halaman *Points* akun FPL Anda, misalnya `https://fantasy.premierleague.com/entry/104821/event/10` -> ID = `104821`), pilih horizon waktu (**1 GW**, **5 GW**, atau **12 GW**), lalu klik **Jalankan Optimasi MILP**.
