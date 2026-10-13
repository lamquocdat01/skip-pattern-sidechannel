# PREREG_39 — Đề tài 39 (B6) · Pre-registration P1 (ghi 09-10-2026, TRƯỚC khi chạy E2)

Khoá lúc ghi file này. **Sau khi ghi không đổi lưới, tầng báo cáo, bộ ước lượng hay tiêu chí CP-B1.** Mọi sai lệch
(bug fix, thiếu dữ liệu) ghi ở §9 "Nhật ký sai lệch" kèm ngày + lý do, và báo ở CP-B1.
Seed toàn cục: **42** (numpy `default_rng(42)`; mỗi khối con dùng `SeedSequence(42).spawn`).

## 0. Hai sự thật dữ liệu ghi nhận trước khi khoá (kiểm 09-10, không phải kết quả E2)
1. **Gate triển khai replay được chính xác.** P0 viết "hysteresis ASMAG không replay được" — sai. Luật
   hysteresis (mở khi S ≥ 0,55; đóng khi S ≤ 0,30; giữ nguyên ở giữa) + watchdog (đóng ≥ 12 khung thì mở mỗi 6 khung)
   + refresh 15 + reuse cap cho đúng `n_opens_deployed` của **cả 121/121 video** (ρ = 0,7695 = vết triển khai).
   Họ gate `deployed` dưới đây là bản replay này. Replay ngưỡng thuần (không hysteresis) cho ρ = 0,709 (= tau_sweep 0,711).
2. **Certificate của reuse cap triển khai là L = 6, không phải 5.** Luật triển khai `fsl > W_maxskip` (W_maxskip = 4)
   cho phép 5 khung bỏ liên tiếp → chu kỳ 6 → mọi sự kiện có lần mở trong [τ₀, τ₀ + 5]. Luật "fixed" `fsl ≥ 4` cho L = 5.
   Bài dùng L = 5 cho họ RR (sàn chu kỳ P = 5) và ghi rõ L = 6 cho gate triển khai.

## 1. Ba tầng báo cáo (quyết định CP-B0 #1 — tầng báo cáo, KHÔNG phải thước đo cổng)
Mọi bảng kết quả báo đủ 3 tầng:
- **all** — mọi video của corpus (121; VIRAT 8 clip có annotation; synthetic 30 video/π).
- **informative** — video có tỷ lệ mở của gate `deployed` (replay) < 0,6 (định nghĩa gốc PLAN §7).
- **evaluable** — xét theo từng bộ (video, W, mục tiêu): A1/A2-cửa-sổ: H(E_W) ≥ 0,5 bit **và** ≥ 30 cửa sổ;
  A3: độ lệch chuẩn của tỷ lệ chiếm dụng theo cửa sổ ≥ 0,1 **và** ≥ 30 cửa sổ; A2-định-vị: video có ≥ 3 onset cô lập (§3).
Tầng được tính trên E (ground truth), không bao giờ trên kết quả tấn công.

## 2. Corpus
| Corpus | Nguồn | E_t (ground truth) | fps | Vai trò |
|---|---|---|---|---|
| C121 | `frame_scores.parquet` (CDnet 53 + LASIESTA 48 + BMC 20), qua `config.yaml`/env | `g_t` | 25 (giả định, khai) | stress case π ≈ 0,77 |
| VIRAT | `THS_DATASETS/VIRAT/videos` 10 clip; 8 clip có annotation DIVA (gitlab.kitware.com/viratdata/viratannotations, KPF) | **E^disc**: hợp các activity *rời rạc* {Opening, Closing, Entering, Exiting, Open_Trunk, Closing_Trunk, Loading, Unloading, PickUp, SetDown, Push, vehicle_starting, vehicle_stopping, vehicle_turning_left, vehicle_turning_right, vehicle_u_turn, activity_running, Riding, Person_Person_Interaction}. Phụ: **E^veh** = hợp mọi activity vehicle_* | 30 (container) | dữ liệu thật, π thấp hơn C121 |
| SYN | sinh từ mô hình (§2.1) | E_t sinh ra | 25 | headline π nhỏ; kiểm Thm 2 |

Score VIRAT: `ASMAGPlusGate.process` (import read-only từ ASMAG src như `generate_gate_scores.py`, resize 320 px),
feedback temporal-persistence lấy "có detection" = có ≥ 1 box annotation (geom) trong khung — **proxy cho YOLO, khai
rõ**. Một worker, dump → `THS_DATASETS/_derived/39/virat/`. Clip không có annotation (2/10) bị loại.

### 2.1 Mô hình synthetic (là MÔ HÌNH, không phải dữ liệu)
- π ∈ {0,02; 0,05; 0,1; 0,2; 0,5} (π = tỷ lệ khung trong sự kiện). Mỗi π: 30 video × T = 15 000 khung (10 phút, 25 fps).
- Độ dài sự kiện D ~ phân bố thực nghiệm `events_gated.csv` (C121, lấy mẫu có hoàn lại); khoảng nghỉ ~ hình học
  với trung bình mean(D)·(1−π)/π (≥ 1 khung).
- Score: copula Gauss AR(1): z_t = φ z_{t−1} + √(1−φ²) ε_t, S_t = F_{E_t}^{-1}(Φ(z_t)) với F₀, F₁ = phân bố thực nghiệm
  của `score` ngoài/trong sự kiện trên C121; φ = tự tương quan lag-1 của normal-score trong các run (ước lượng một lần,
  ghi vào `results/P1_synthetic_model.json`).

## 3. Mục tiêu tấn công
- **A1** presence: E_W = 1[∃ t ∈ W: E_t = 1].
- **A2** onset: (i) cửa sổ: E_W^on = 1[∃ onset ∈ W]; (ii) **định vị**: với mỗi onset *cô lập* (≥ 1 s không sự kiện
  trước nó), bộ ước lượng trả τ̂₀; báo MAE (khung, lỗi cắt ở 2 s nếu không phát hiện) và Pr[|τ̂₀ − τ₀| ≤ L], L = 5.
  Bộ ước lượng: (a) **CUSUM** trên chuỗi mở ước lượng (tốc độ ρ_lo → ρ_hi học trên train), τ̂₀ = điểm bắt đầu của
  đoạn CUSUM dương cuối trước báo động; (b) **matched filter** bậc thang độ dài 1 s, τ̂₀ = argmax tương quan trong
  [τ₀ − 2 s, τ₀ + 2 s]. Ngưỡng CUSUM chọn trên train để tỷ lệ báo động giả ≤ 1/phút.
- **A3** activity: hồi quy theo cửa sổ W: y₁ = tỷ lệ chiếm dụng (mean E_t), y₂ = số onset/phút. Báo R²(y₁),
  MAE(y₂) (sự kiện/phút).

## 4. Kênh, mức quan sát — 12 ô = 2 kênh × 2 mức × 3 mục tiêu
- **power oracle** = **traffic oracle** = a_t chính xác (tương đương thông tin; báo một lần, ghi chú).
- **power noisy (headline)**: f_s = 1 Hz, σ = 0,05 W, lượng tử 0,01 W, P_base = 5 W, năng lượng/khung theo r = E_active/E_idle
  (instance Deep3 C2b 12,5/0,046 mJ, r ≈ 272, khai B9). Quét phụ: f_s ∈ {1,2,5,10} Hz × σ ∈ {0; 0,02; 0,05; 0,1} W.
- **traffic noisy (headline)**: một burst/lần mở, jitter thời gian N(0, 50 ms), kích thước N(1200, 150) B; quan sát viên
  chỉ giữ thời điểm. Quét phụ: jitter ∈ {5, 50, 200} ms.
- Cửa sổ W ∈ {1, 2, 5, 10} s (không chồng lấn).

## 5. Họ gate và lưới (khoá)
| Mã | Họ | Lưới |
|---|---|---|
| G1 | content-aware + shield (refresh 15, reuse cap triển khai, L = 6) | τ ∈ {0,05; 0,20; 0,35; 0,45; 0,55; 0,65; 0,72; 0,76; 0,78; 0,80; 0,83} |
| G2 | content-aware thuần (S_t > τ) | cùng lưới τ |
| G3 | deployed (replay hysteresis 0,55/0,30 + watchdog 12/6 + refresh 15 + reuse cap) | 1 cấu hình |
| G4 | periodic, pha ngẫu nhiên | P ∈ {2, 3, 5, 8, 15, 30} |
| G5 | rate-limited (τ = 0,55, W_b = 25, sàn P = 5) | K ∈ {3, 5, 8, 12} |
| G6 | OR-dummy (P0 "privacy_certified", p₁ = 1, ε = ∞) — đối chứng | λ ∈ {0,05; 0,1; 0,2; 0,3} (K = 5) |
| G7 | **RR-certified** (Thm 3): a_t = F_t ∨ RR(c_t), sàn P = L = 5, τ = 0,55, W_b = 25 | K ∈ {5, ∞} × p₁ ∈ {0,5; 0,7; 0,9} × λ ∈ {0,02; 0,05; 0,1; 0,2} (24 cấu hình) |
| G8 | (chỉ kiểm Thm 2, SYN) content-aware thuần ∨ refresh P_r, không reuse cap | P_r ∈ {8, 15, 30}, τ ∈ {0,35; 0,55; 0,72} |
Gate RR: c_t = demand đã qua token bucket; nếu F_t = 1 → mở; ngược lại mở với xác suất p₁ nếu c_t = 1, λ nếu c_t = 0.

**Tập "core"** chạy tấn công classifier đầy đủ (12 ô × 4 W × 3 tầng × 3 corpus): G1 cả lưới τ (cho hình ∩), G3, G4 P = 5,
G5 K = 5, G6 λ = 0,2, G7 {K = 5, p₁ = 0,7, λ = 0,1} và {K = ∞, p₁ = 0,5, λ = 0,2}. MI plug-in oracle chạy trên **toàn lưới**.
Quét quan sát viên phụ (f_s × σ, jitter) chỉ cho G1 τ = 0,55, G3, G5 K = 5, G7 {K = 5, p₁ = 0,7, λ = 0,1}, W = 1 s và 5 s.

## 6. Bộ ước lượng MI (3 cách) và CI
1. **Plug-in binned** (như P0): O = N_W chia ≤ 8 bin đều; hiệu chỉnh bias bằng trung bình permutation null 200 lần
   (xáo E giữa các cửa sổ trong video); p = tỷ lệ null ≥ quan sát.
2. **KSG-type kNN** (Ross 2014, target rời rạc – đặc trưng liên tục; `sklearn.feature_selection.mutual_info_classif`,
   k = 3) trên N_W chuẩn hoá + rung 1e-6.
3. **Cận dưới classifier**: I ≥ H(E_W) − CE_test (bit), CE = cross-entropy của GBM trên video test (out-of-fold);
   cắt ở 0. Hợp lệ là cận dưới cho bất kỳ classifier nào.
CI: bootstrap 500 lần theo video (resample video có hoàn lại). Gộp theo tầng = trung bình theo video.

## 7. Tấn công — huấn luyện / kiểm tra
- Đặc trưng cửa sổ (cố định): số lần mở/khung; số mở trong 5 bin con; khoảng trống dài nhất (tỷ lệ W); số run;
  vị trí mở đầu/cuối (tỷ lệ W); số mở cửa sổ trước và sau. Kênh noisy dùng số mở ước lượng (power: nghịch đảo mô hình
  công suất theo mẫu; traffic: đếm burst theo bin).
- Classifier: logistic (chuẩn hoá, C = 1) và GBM (`HistGradientBoostingClassifier`, mặc định, max_iter = 200).
  **GBM là chính** cho tiêu chí; logistic báo kèm. Hồi quy A3: Ridge và `HistGradientBoostingRegressor`.
- Chia theo video: GroupKFold 5 (xáo nhóm, seed 42); kết quả out-of-fold, AUC gộp theo tầng + CI bootstrap theo video.
- **Attacker không biết τ**: với G1/G2, mô hình kiểm trên τ_test được huấn luyện trên lịch của *mọi τ khác* trong lưới
  (video train). G3 (deployed) dùng mô hình huấn luyện trên G1 mọi τ ≠ 0,55. Họ khác: attacker biết tham số công bố
  của defense (Kerckhoffs) — huấn luyện trên cùng cấu hình.
- **Cross-dataset** (A1, oracle + 2 noisy headline, G1 τ = 0,55): CDnet → LASIESTA; CDnet+LASIESTA → BMC;
  C121 → VIRAT; đặc trưng theo đơn vị thời gian (đã chuẩn hoá theo fps).

## 8. Kiểm cận vs số (T10) và tiêu chí CP-B1 (mặc định)
- Thm 1(c): I(E; N_W) đo so với log₂(K+1) (G5, G7 K = 5) — kỳ vọng cận lỏng ở A1.
- Thm 2: SYN khối thẳng hàng độ dài L = 5 (onset ở đầu khối với xác suất π_b, D ≥ L), gate G8 và G4 P ∈ {5, 8};
  đo p̂₁ = Pr[B=1|E=1], p̂₀, ρ̂₀, I(E;B) so với f(π_b, Lρ̂₀); lịch chu kỳ P = L phải cho I = 0 tại ρ₀ = 1/L.
- Thm 3: quét G7 → (ρ, ε lý thuyết, I đo); so iso-energy với periodic bằng trễ trung bình và tỷ lệ miss sự kiện D < L.

**Tiêu chí CP-B1 (default, khoá):**
1. Thm 2 (aligned) + Hệ quả 2.1/2.2 có chứng minh đầy đủ trong `theory/`, và trên SYN: I(E;B) đo ≥ f(π_b, Lρ̂₀) − CI
   ở mọi cấu hình có certificate (p̂₁ = 1); P = L cho I ≈ 0 (|I| < 0,01 bit).
2. A1 **AUC > 0,8** (GBM, τ-agnostic, G1 τ = 0,55, W = 1 s) ở **ít nhất một kênh noisy headline** trên tầng evaluable
   của C121; báo cả hai kênh, cả 3 tầng, và G3.
3. **SYN π ≤ 0,1**: MI plug-in hiệu chỉnh (A1, oracle, G1 τ = 0,55, tầng all) > 0,2 bit/cửa sổ ở **W = 10 s**
   (W lớn nhất, vì H(E_W) với π nhỏ tăng theo W). Nếu trần H(E_W) < 0,2 bit thì tiêu chí này *không thể đạt về nguyên tắc*
   — báo "không đạt (trần entropy)" kèm MI chuẩn hoá MI/H(E_W), không đổi ngưỡng. VIRAT báo cùng thước đo, không phải
   điều kiện (π_VIRAT ≈ 0,2–0,5).

## 9. Nhật ký sai lệch (điền khi chạy)
- (trống lúc khoá)
- 09-10 (trước khi chạy E2, làm rõ cài đặt — không đổi lưới): (1) G7 dùng token bucket **tumbling** (K demand đầu tiên
  trên mỗi cửa sổ lưới W_b), vì Thm 3 cần Δ bị chặn; bucket trượt cho Δ không bị chặn (phản ví dụ trong
  `theory/P1_theorems.md` §4). G5 giữ bucket trượt như P0. (2) "Báo động giả" của CUSUM = báo động khi không có khung sự
  kiện nào trong 2 s tính đến thời điểm báo động. (3) Bỏ qua các luồng A2-định-vị lỗi cắt 2 s = 2·fps khung. (4) Pha sàn
  của G5 lấy ngẫu nhiên độc lập (P0 cố định pha 0). (5) T10 thêm gate kiểm **G8\*** (oracle-certified: mở đúng khung onset
  + Bernoulli(q) ngoài sự kiện, q ∈ {0; 0,02; 0,05; 0,1; 0,15; 0,19}) vì G8 với score synthetic có p̂₁ < 1 nên tiêu chí 1
  sẽ rỗng nếu chỉ dùng G8; G8\* chỉ dùng kiểm định lý, không vào bảng tấn công.
- 09-10 (sau khi chạy, không đổi tiêu chí): T10 Thm 3 chỉ chạy trên SYN π 0,05 / 0,1, C121, VIRAT (đúng như cài đặt trước khi chạy trong `verify_bounds.py`); SYN π 0,5 ở W = 10 s có H(E_W) = 0 (suy biến). Kết quả tiêu chí: 1 ĐẠT, 2 KHÔNG ĐẠT, 3 KHÔNG ĐẠT (không phải trần entropy) — xem `CP_B1_39_20261105.md`.
- 09-10 **P1b (POST-HOC sau CP-B1, quyết định A′)** — không thay tiêu chí nào, ngưỡng §8 giữ nguyên:
  (T13) `code/rho_sweep.py`: quét ρ ∈ {0,05…0,40} (τ tính ngược theo corpus) cho G2 thuần và G8 refresh P_r ∈ {5, 10, 20}
  so với periodic cùng ρ theo clip, trên VIRAT / C121 all / C121 evaluable. Mục đích **chỉ** là xác định vùng vận hành;
  điểm có tỷ lệ demand < 5 % ρ (gate ≡ periodic) đánh dấu "degenerate" và không tính là thắng (quy tắc thêm SAU lần chạy
  đầu, khi thấy τ vượt max score ở ρ thấp nhất; lần chạy đầu có 1 "thắng" tại điểm degenerate — C121_eval G8_Pr20 ρ 0,05
  miss D ≥ L — bị loại bởi quy tắc này; số thắng còn lại không đổi). Không hiệu chỉnh đa so sánh (~100 ô).
  (T14) mô hình SYN v2 post-hoc: phân bố score ngoài sự kiện fit từ khung không-sự-kiện của VIRAT (`data_sparse.py v2`,
  `results/P1b_synthetic_model_v2.json`); MI oracle chạy lại `code/syn_v2.py` → `results/P1b_syn_v2_mi.csv`, Thm 2 chạy
  lại `verify_bounds.py thm2v2` → `results/P1b_bounds_thm2_synv2.csv`. **Tiêu chí 3 bản khoá (v1) vẫn KHÔNG ĐẠT.**
- 09-10 **P2 (HELD)**: sửa tái lập — `verify_bounds.py` bỏ `hash(lab)` (phụ thuộc PYTHONHASHSEED), seed tường minh `--seed 42`;
  bảng Thm 2/Thm 3 sinh lại, không kết luận nào đổi (`results/P2_reseed_diff.md`); so sánh iso-energy ở Thm 3 báo dạng khoảng
  qua seed 42–46 (`results/P2_thm3_seed_sweep.csv`). Không đổi tiêu chí/ngưỡng.
- 13-10 **P3 T22b — real-plug validation (POST HOC, sau mọi tiêu chí; không đổi tiêu chí/ngưỡng nào)**: lịch G3 của 3 clip
  (CDnet2014/blizzard, LASIESTA/O_RA_02, VIRAT_S_000002) phát lại trên ESP32-S3 (khung mở = 72 ms tính toán, khung đóng = idle,
  12,5 khung/s ⇒ trục thời gian dãn 2,0×/2,4×), đo dòng toàn bo bằng INA219 nối tiếp nguồn 5 V, 10 Hz
  (`code/real_trace/`, dump `results/real_trace/raw/20261013T161503Z/`). Phân tích `align.py` + `attack_real.py`: cùng đặc
  trưng/target/classifier của §7 nhưng **giao thức chia khác §7** (không có đủ video thật cho GroupKFold): within-video
  (nửa đầu train, nửa sau test) và cross-video blizzard ↔ VIRAT; attacker huấn luyện trên chính lịch G3 (không τ-agnostic).
  Quan sát: oracle, plug mô phỏng headline (§4), plug thật trung bình mỗi giây video (đối chứng chính), plug thật 10 Hz
  (cận trên, lợi do replay chậm). U mở rộng k = 2 = 2·√(u_boot² + u_seed²), bootstrap khối 10 s × 5 seed; MI plug-in
  hiệu chỉnh null 200. Quy tắc thêm SAU lần chạy đầu: dự đoán hằng (HGB không tách được với 24 cửa sổ train ở W = 5 s)
  ghi "không đánh giá" thay vì AUC 0,5; ở W = 5 s báo attacker logistic. LASIESTA (375 khung) chỉ là smoke test.
