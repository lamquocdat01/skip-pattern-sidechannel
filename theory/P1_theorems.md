# Đề tài 39 — Định lý và chứng minh (P1, 09-10-2026) · LaTeX-ready

Toán viết bằng `$…$` để chép thẳng sang LaTeX (P3). Ký hiệu khớp `theory_notes.md` và `code/`.
Trạng thái từng kết quả ở bảng §0. Kiểm số (T10) ở `code/verify_bounds.py`, `results/P1_bounds_*.csv`.

## 0. Trạng thái
| Kết quả | Trạng thái P1 | Kiểm số |
|---|---|---|
| Thm 1 (a)–(c) | chứng minh đủ | (c) vs $I(E;N_W)$ đo |
| Thm 1 (d) | chứng minh đủ cho lớp "gate chuyển mode"; điều kiện O(log T) nêu rõ | — (định tính) |
| Thm 2 (aligned) — **định lý chính** | chứng minh đủ (lập luận *khối chọn ngẫu nhiên đều*, không cần dừng) | SYN khối thẳng hàng |
| Ngưỡng $1/L$ chặt | chứng minh đủ | periodic $P=L$ |
| Hệ quả 2.1 (onset bất kỳ) | chứng minh đủ; **hệ số 2 gỡ được cho câu hỏi "có sự kiện tại thời điểm ứng viên $\tau_0$"** (pha ngẫu nhiên); với câu hỏi "có onset trong cửa sổ $m$ khung" hệ số là $(m+L-1)/L$, $m=L$ cho $2-1/L$ — không gỡ được, giữ và nói rõ | SYN |
| Hệ quả 2.2 (không $\varepsilon$ hữu hạn; mạnh hơn: $\delta \ge 1-L\rho_0$) | chứng minh đủ | — |
| Mệnh đề 2.3 (A2, định vị onset) | chứng minh đủ (dạng entropy + xác suất định vị) | A2 trên SYN |
| Thm 3 (i)–(iv) RR | chứng minh đủ; $\Delta$ tính đúng cho bucket **tumbling**; sliding bucket cho $\Delta$ không bị chặn (phản ví dụ) | quét G7 |
| Thm 3 (v) Pareto $K=\infty$ | dạng đóng | Pareto đo |
| Liên hệ Cuff–Yu | phát biểu §5, DOI 10.1145/2976749.2978308 ✓ | — |

---

## 1. Mô hình

Khung $t\in[T]=\{1,\dots,T\}$. Chuỗi sự kiện $E=(E_1,\dots,E_T)\in\{0,1\}^T$; một *sự kiện* là một run cực đại của $E_t=1$ với
onset $\tau_0$ và độ dài $D$. Điểm nội dung $S_t$; trường hợp xấu nhất (dùng cho mọi cận trên): $S=s(E)$ là hàm tất định
của $E$. Gate nhân quả: $a_t=g_t(S_{1:t},a_{1:t-1},U)$, $U$ là ngẫu nhiên riêng của gate, $U\perp E$. Quan sát $O=h(A,N)$,
$N\perp(E,U)$. Vì $E\to(S,U)\to A\to O$ là chuỗi Markov:
$$I(E;O)\le I(E;A). \tag{DPI}$$

**Def (certificate $L$).** Với mọi $e$ và mọi sự kiện $(\tau_0,D)$ với $D\ge L$: $\Pr[\exists t\in[\tau_0,\tau_0+L-1]: a_t=1]=1$.

**Def ($\rho_0$).** Với thế giới *yên tĩnh* $e^{\varnothing}$ (không sự kiện trên đoạn xét), $\rho_0=\frac1T\mathbb E[\sum_t a_t\mid e^{\varnothing}]$.
Với tiên nghiệm có sự kiện, $\rho_0$ = tỷ số kỳ vọng (số lần mở trong khung không sự kiện)/(số khung không sự kiện).

**Def ($(\varepsilon,\delta)$-schedule-privacy).** $e\sim e'$ nếu khác nhau đúng một sự kiện có $D\le D_{\max}$. Gate là
$(\varepsilon,\delta)$-schedule-private nếu $\Pr[A\in\mathcal S\mid e]\le e^{\varepsilon}\Pr[A\in\mathcal S\mid e']+\delta$ với mọi $e\sim e'$, mọi $\mathcal S$.

Ký hiệu $h_2(x)=-x\log_2x-(1-x)\log_2(1-x)$.

---

## 2. Thm 1 — Cận đếm lịch

**Giả thiết.** $a_t=F_t\vee C_t$ với $F$ là lịch *oblivious* ($F=\phi(U_F)$, $U_F\perp E$) và $C$ là tập lần mở nội dung
(chỉ ở khung $F_t=0$), thoả: mọi cửa sổ cố định $W$ khung chứa nhiều nhất $K_W$ lần mở nội dung.
(Token bucket $\le K$ trên mỗi $W_b$ khung trượt cho $K_W\le K\lceil W/W_b\rceil$.)

**Thm 1.**
(a) Nếu $C\equiv0$ (chu kỳ, pha cố định hoặc ngẫu nhiên độc lập $E$): $I(E;A)=0$.
(b) $I(E;A_W)\le\log_2\sum_{k=0}^{K_W}\binom{W}{k}\le W\,h_2(K_W/W)$ khi $K_W\le W/2$.
(c) Với quan sát viên chỉ đếm $N_W=\sum_{t\in W}a_t$: $I(E;N_W)\le\log_2(K_W+1)$.
(d) *Gate chuyển mode.* Giả sử $A$ là hàm tất định của $(U, \sigma)$ với $\sigma=(s_1<\dots<s_M)$ là dãy thời điểm chuyển
mode, $M\le M_T$ hầu chắc chắn. Khi đó $I(E;A)\le\log_2\sum_{m\le M_T}\binom{T}{m}\le M_T\log_2(eT/M_T)$.
Hệ quả: nếu mode chỉ chuyển *đơn điệu* giữa $P_{\max}$ và $P_{\min}$ bằng cách chia đôi chu kỳ thì
$M_T\le\lceil\log_2(P_{\max}/P_{\min})\rceil$ (hằng số) nên $L_T=O(\log T)$; nếu mode được *đặt lại* sau mỗi epoch
và số epoch là $O(\log T)$ (epoch nhân đôi kiểu Askarov–Zhang–Myers) thì $L_T=O(\log^2 T)$.

*Chứng minh.* (a) $A=\phi(U_F)\perp E$.
(b) $I(E;A_W)\le I(E;A_W,F_W)=I(E;F_W)+I(E;A_W\mid F_W)=0+I(E;A_W\mid F_W)\le H(A_W\mid F_W)$. Cho trước $F_W$,
$A_W$ xác định bởi $C_W$, là tập con của $W$ có $\le K_W$ phần tử, nên $H(A_W\mid F_W)\le\log_2\sum_{k\le K_W}\binom Wk$.
Cận $\sum_{k\le\alpha n}\binom nk\le2^{nh_2(\alpha)}$ với $\alpha\le1/2$ là chuẩn.
(c) $N_W=|F_W|+|C_W|$ vì $C$ chỉ ở khung $F_t=0$. Như (b): $I(E;N_W)\le I(E;N_W\mid F_W)\le H(|C_W|)\le\log_2(K_W+1)$.
(d) $I(E;A)\le I(E;A,U)=I(E;A\mid U)\le H(\sigma\mid U)\le\log_2|\{\sigma\}|$, và số dãy tăng độ dài $\le M_T$ trong $[T]$
là $\sum_{m\le M_T}\binom Tm\le (eT/M_T)^{M_T}$. Với chia đôi đơn điệu, mỗi lần chuyển giảm $\log_2P$ ít nhất 1, nên
$M_T\le\log_2(P_{\max}/P_{\min})$. Với epoch đặt lại, $M_T=O(\log T)$ cho $O(\log^2T)$. $\square$

*Đối chiếu Askarov–Zhang–Myers (đọc từ PDF tác giả, xem `related_work_boundary.md`).* CCS'10 "fast doubling": quantum nhân
đôi mỗi lần đoán sai ⇒ $\le\log(T+1)+1$ epoch ⇒ rò rỉ $\le\log^2(T+1)+\log(T+1)=O(\log^2T)$ — **không phải** $O(\log T)$;
$O(\log T)$ chỉ đạt khi hoãn epoch mới qua "deferral point" (trả giá thông lượng/trễ). CCS'11: $O(\log T\cdot\log M)$, và
$(\log T_w+2)\log(M+1)$ khi có thời gian thực thi xấu nhất $T_w$. Bản (d) của ta là tương tự trực tiếp: với certificate $L$,
mọi mode có $P\le L$; nếu mode chỉ chuyển đơn điệu ($P_{\max}=L\to P_{\min}\ge1$) thì $M_T\le\lceil\log_2L\rceil$ và
$L_T\le\lceil\log_2L\rceil\log_2(eT)=O(\log L\cdot\log T)$ — vai trò của $L$ giống $T_w$. Chuyển mode không đơn điệu (đặt lại sau
khi sự kiện kết thúc, cần cho năng lượng) đưa về $M_T$ = số lần đặt lại, tức trường hợp $O(\log^2T)$ hoặc tuyến tính.
PLAN §2 viết "leakage $O(\log T)$" là **sai** cho gate nhân đôi; bài phải viết $O(\log^2T)$ (epoch nhân đôi) hoặc
$O(\log L\log T)$ (đơn điệu, có certificate).

**Nhận xét (khớp số đo).** Với mục tiêu nhị phân $H(E_W)\le1$, nên (b)/(c) vô dụng ngay khi $\log_2(K_W+1)\ge1$, tức $K_W\ge1$:
rate limiting thuần không bảo vệ A1. (c) có ích khi mục tiêu nhiều bit (A2 vị trí onset, A3 số sự kiện) và cho toàn vết.
Điều kiện (d) là điều kiện cấu trúc: lịch phải được *quyết định bởi ít sự kiện chuyển mode*; gate content-aware thuần
có $M_T$ tỉ lệ với $T$ nên cận (d) tuyến tính theo $T$.

---

## 3. Thm 2 — Trilemma Latency–Energy–Privacy

**Thiết lập aligned.** Chia $[T]$ thành $M=T/L$ khối $\mathcal B_j=[jL,(j+1)L)$. Tiên nghiệm (bất kỳ phụ thuộc giữa các
khối) sao cho mỗi khối hoặc *bắt đầu* một sự kiện có $D\ge L$ đúng ở đầu khối ($Z_j=1$), hoặc không chứa khung sự kiện nào
($Z_j=0$); có thể có khối loại thứ ba (đang giữa sự kiện) — chúng không dùng trong phép thử. Chọn $J$ đều trên các khối,
độc lập với mọi thứ khác. Đặt $\pi=\Pr[Z_J=1\mid Z_J\in\{0,1\}]$, $B=\mathbb 1[\exists t\in\mathcal B_J:a_t=1]$, và
$\rho_0=\mathbb E[\#\text{mở trong khối }Z=0]/\mathbb E[\#\text{khung trong khối }Z=0]$.

**Thm 2 (aligned).** Nếu gate có certificate $L$ và $L\rho_0<1$ thì
$$I(Z_J;A,J)\;\ge\;I(Z_J;B)\;\ge\;f(\pi,L\rho_0):=h_2\big(\pi+(1-\pi)L\rho_0\big)-(1-\pi)\,h_2(L\rho_0)\;>\;0,$$
(điều kiện trên $Z_J\in\{0,1\}$). Tương đương dạng Le Cam: $\mathrm{TV}(P_{B\mid Z=1},P_{B\mid Z=0})\ge1-L\rho_0$, và phép thử
"báo sự kiện iff $B=1$" có miss $=0$, báo động giả $\le L\rho_0$.

*Chứng minh.* (1) Certificate: khối $Z_J=1$ chứa $[\tau_0,\tau_0+L-1]$ nên $p_1:=\Pr[B=1\mid Z_J=1]=1$.
(2) Đếm: trong một thực hiện, số khối $Z=0$ có ít nhất một lần mở $\le$ số lần mở trong các khối $Z=0$. Lấy kỳ vọng và
chia cho $\mathbb E[\#\text{khối }Z=0]=\mathbb E[\#\text{khung }Z=0]/L$:
$p_0:=\Pr[B=1\mid Z_J=0]=\frac{\mathbb E[\#\{j:Z_j=0,B_j=1\}]}{\mathbb E[\#\{j:Z_j=0\}]}\le\frac{\mathbb E[\#\text{mở trong khối }Z=0]}{\mathbb E[\#\text{khung }Z=0]/L}=L\rho_0.$
(Đẳng thức đầu dùng $J$ đều và độc lập — đây là chỗ thay cho giả thiết dừng.)
(3) Với $p_1=1$: $I(Z;B)=H(B)-H(B\mid Z)=h_2(\pi+(1-\pi)p_0)-(1-\pi)h_2(p_0)=:g(p_0)$.
(4) $g'(p)=(1-\pi)\big[\log_2\tfrac{1-q}{q}-\log_2\tfrac{1-p}{p}\big]$ với $q=\pi+(1-\pi)p>p$, nên $g'<0$ trên $(0,1)$:
$g$ giảm ngặt, $g(1)=0$. Vậy $p_0\le L\rho_0<1\Rightarrow I(Z;B)\ge g(L\rho_0)=f(\pi,L\rho_0)>0$.
(5) $B$ là hàm của $(A,J)$ nên $I(Z_J;A,J)\ge I(Z_J;B)$; TV $=|p_1-p_0|=1-p_0\ge1-L\rho_0$. $\square$

**Ngưỡng $1/L$ chặt.** Lịch chu kỳ $P=L$ (pha bất kỳ độc lập $E$) có certificate $L$ (mọi $L$ khung liên tiếp chứa một lần mở),
$\rho_0=1/L$ và $I(E;A)=0$ (Thm 1a). Hơn nữa $f(\pi,x)\to0$ liên tục khi $x\uparrow1$. Vậy biên $\{\rho_0<1/L\}$ là chính xác:
không gate nào có certificate $L$, $\rho_0<1/L$ và rò rỉ 0; với $\rho_0\ge1/L$ thì có. $\square$

**Bản tổng quát $p_1<1$** (dùng cho kiểm số khi certificate chỉ đúng một phần):
$I(Z;B)=h_2(\pi p_1+(1-\pi)p_0)-\pi h_2(p_1)-(1-\pi)h_2(p_0)$.

### Hệ quả 2.1 — onset bất kỳ (thử gỡ hệ số 2, quyết định CP-B0 #4)
(a) **Thời điểm ứng viên ngẫu nhiên (gỡ được hệ số 2).** Thế giới yên tĩnh $H_0$ hoặc $H_1$ = một sự kiện $D\ge L$ bắt đầu
tại $\tau_0$, với $\tau_0$ đều trên $\{1,\dots,T-L+1\}$, độc lập gate, và *attacker biết $\tau_0$* (câu hỏi "có ai vào lúc 8:00
không?"). Với $B=\mathbb 1[\exists t\in[\tau_0,\tau_0+L-1]:a_t=1]$ và $\Pr[H_1]=\pi$:
$$I(Z;A,\tau_0)\ge f\big(\pi,\;L\rho_0\cdot\tfrac{T}{T-L+1}\big).$$
*Chứng minh.* Dưới $H_1$, $B=1$ (certificate). Dưới $H_0$: mỗi lần mở tại $s$ làm $B=1$ cho nhiều nhất $L$ giá trị
$\tau_0\in[s-L+1,s]$, nên $\Pr[B=1\mid H_0]=\frac{1}{T-L+1}\mathbb E\sum_{\tau}\mathbb 1[\cdot]\le\frac{L\,\mathbb E[N_T]}{T-L+1}=L\rho_0\frac{T}{T-L+1}$.
Phần còn lại như Thm 2. $\square$
Đây chính là "pha khối ngẫu nhiên": khối độ dài $L$ đặt tại vị trí đều; hệ số chỉ còn $T/(T-L+1)\to1$.

(b) **Onset chưa biết trong cửa sổ $m$ khung.** Nếu attacker chỉ biết onset nằm trong một cửa sổ $m$ khung đã biết vị trí
(câu hỏi A2-cửa-sổ), phép thử dùng khối $m+L-1$ khung và cho $I\ge f(\pi_m,(m+L-1)\rho_0)$, hiệu lực khi $\rho_0<1/(m+L-1)$.
Với $m=L$: hệ số $(2L-1)/L=2-1/L$. *Không gỡ được* bằng pha ngẫu nhiên: certificate chỉ ràng buộc khoảng $[\tau_0,\tau_0+L-1]$
gắn với $\tau_0$, nên khi $\tau_0$ chưa biết trong $m$ khung thì vùng phải quan sát dài $m+L-1$. Ta giữ hệ số này và nói rõ.
Kết luận định tính (ngưỡng $1/L$ cho *rò rỉ 0*) không phụ thuộc hệ số: theo (a), với $\rho_0<1/L$ luôn tồn tại thời điểm
ứng viên có rò rỉ dương.

### Hệ quả 2.2 — không có $\varepsilon$ hữu hạn
Nếu gate có certificate $L$ và $L\rho_0<1$ (trong thế giới yên tĩnh) thì với mọi $\varepsilon<\infty$ gate **không**
$(\varepsilon,\delta)$-schedule-private với bất kỳ $\delta<1-L\rho_0\frac{T}{T-L+1}$; đặc biệt không $\varepsilon$-private ($\delta=0$).
*Chứng minh.* Theo (a), $\frac1{T-L+1}\sum_{\tau}\Pr[B_\tau=1\mid e^{\varnothing}]\le L\rho_0\frac{T}{T-L+1}<1$, nên tồn tại $\tau^*$ với
$\Pr[B_{\tau^*}=0\mid e^{\varnothing}]\ge1-L\rho_0\frac{T}{T-L+1}$. Lấy $e'$ = $e^{\varnothing}$ thêm một sự kiện $D=L$ tại $\tau^*$ ($e\sim e'$),
$\mathcal S=\{B_{\tau^*}=0\}$: $\Pr[\mathcal S\mid e']=0$, nên $\Pr[\mathcal S\mid e^{\varnothing}]\le e^{\varepsilon}\cdot0+\delta$ buộc
$\delta\ge1-L\rho_0\frac{T}{T-L+1}$. $\square$
Hệ quả cho thiết kế: muốn $\varepsilon$ hữu hạn với certificate $L$ thì sàn $\rho_0\ge1/L$ là bắt buộc — chính là sàn chu kỳ
$P=L$ của Thm 3.

### Mệnh đề 2.3 — nhánh A2 (định vị onset)
Thế giới $H_1$: một sự kiện $D\ge L$, $\tau_0$ đều trên $T$ khung, phần còn lại yên tĩnh. Gọi $N$ là tổng số lần mở,
$\bar n_{\mathrm{ev}}=\mathbb E[\#\text{mở trong sự kiện}]$. Khi đó
$$H(\tau_0\mid A)\le\log_2\!\big(L\,(\rho_0T+\bar n_{\mathrm{ev}})\big),\qquad I(\tau_0;A)\ge\log_2\frac{1}{L(\rho_0+\bar n_{\mathrm{ev}}/T)}\xrightarrow{T\to\infty}\log_2\frac1{L\rho_0},$$
và attacker chọn một lần mở đều ngẫu nhiên $o$ rồi đặt $\hat\tau=o-\lfloor (L-1)/2\rfloor$ đạt
$\Pr[|\hat\tau-\tau_0|\le\lceil (L-1)/2\rceil]\ge1/(\rho_0T+\bar n_{\mathrm{ev}})$, gấp $\ge\frac{T}{L(\rho_0T+\bar n_{\rm ev})}\to\frac1{L\rho_0}$ lần
đoán mù trong cùng dung sai.
*Chứng minh.* Certificate: $\tau_0\in\mathcal C(A)=\{t:\exists\text{ mở trong }[t,t+L-1]\}$, $|\mathcal C(A)|\le LN$. Cho trước $A$,
$\tau_0$ nằm trong tập $\le LN$ phần tử: $H(\tau_0\mid A)\le\mathbb E\log_2(LN)\le\log_2(L\,\mathbb EN)$ (Jensen). Dưới $H_1$,
$\mathbb EN\le\rho_0T+\bar n_{\rm ev}$ (lần mở ngoài sự kiện có tỷ lệ $\le\rho_0$ theo định nghĩa trên thế giới yên tĩnh; khung
sau sự kiện cùng phân phối do gate nhân quả — giả thiết *mixing*: gate quên sự kiện sau thời gian hữu hạn, chi phí đó
gộp vào $\bar n_{\rm ev}$). $I=\log_2T-H(\tau_0\mid A)$. Với bộ ước lượng: có ít nhất một lần mở trong $[\tau_0,\tau_0+L-1]$,
chọn trúng nó với xác suất $\mathbb E[1/N]\ge1/\mathbb EN$. $\square$
Fano đọc ngược: mọi gate có $\rho_0<1/L$ và certificate $L$ cho attacker $\ge\log_2(1/(L\rho_0))$ bit về $\tau_0$.

---

## 4. Thm 3 — Gate privacy-certified bằng randomized response

**Cấu trúc.** $a_t=F_t\vee R_t$, trong đó
- $F_t$: sàn chu kỳ $P=L$, pha $\phi\sim\mathrm{Unif}\{0..L-1\}$ độc lập;
- $d_t=\mathbb 1[S_t>\tau]$; $c_t$ = demand được nhận bởi **token bucket tumbling**: trên mỗi cửa sổ lưới cố định
  $[kW_b,(k+1)W_b)$ nhận $K$ demand *đầu tiên* (bucket tính trên $d$, *trước* RR, nên $c=c(E)$ tất định);
- $R_t\mid c\sim\mathrm{Bernoulli}(p_1)$ nếu $c_t=1$, $\mathrm{Bernoulli}(\lambda)$ nếu $c_t=0$, độc lập theo $t$; $0<\lambda<p_1<1$.

**Footprint.** Thêm/bớt một sự kiện độ dài $D$ làm $d$ đổi trên tập $\mathcal F$ với $|\mathcal F|\le D+h$, $h$ = bộ nhớ của score
($h=0$ nếu $S_t$ chỉ phụ thuộc $E_t$; ASMAG có `recent_object_memory` = 30 khung).

**Thm 3.**
(i) *Certificate:* mọi sự kiện (mọi $D\ge1$) có lần mở trong $[\tau_0,\tau_0+L-1]$ với xác suất 1.
(ii) *Privacy:* gate là $\varepsilon$-schedule-private với $\varepsilon=\Delta\cdot\varepsilon_0$,
$\varepsilon_0=\max\{\ln(p_1/\lambda),\ln((1-\lambda)/(1-p_1))\}$, và
$$\Delta=\max_{e\sim e'}|\{t:c_t\ne c'_t\}|\le\begin{cases}|\mathcal F|, & K\ge W_b\ (\text{bucket không hoạt động}),\\ \sum_{k}2\min\{K,|\mathcal F\cap w_k|\}\le2\min\{K\,n_w,\;|\mathcal F|\}, & K<W_b,\end{cases}$$
với $w_k$ các cửa sổ lưới cắt $\mathcal F$, $n_w\le\lceil(|\mathcal F|-1)/W_b\rceil+1$. Cận $2\min\{K,|\mathcal F\cap w|\}$ đạt được (chặt).
(iii) *Năng lượng:* $\rho=\frac1L+(1-\frac1L)\big(\lambda+(p_1-\lambda)\bar c\big)$, $\bar c$ = tỷ lệ demand được nhận
trên khung ngoài sàn, $\bar c\le\min\{\bar d,K/W_b\}$.
(iv) *Rò rỉ:* $I(E;A_W)\le(1-\frac1L)\sum_{t\in W}I(c_t;R_t)\le(1-\frac1L)\,|W|\,C_{\rm BAC}(\lambda,p_1)$, với $C_{\rm BAC}$
là dung lượng kênh nhị phân bất đối xứng $\{0\mapsto\lambda,1\mapsto p_1\}$; và (Cuff–Yu, §5) với một sự kiện,
$I(X_i;A\mid X_{-i})\le\min\{\varepsilon,\varepsilon^2\}\log_2e$ bit.
(v) *OR-dummy ($p_1=1$) cho $\varepsilon=\infty$*: $\Pr[a_t=0\mid c_t=1,F_t=0]=0<\Pr[a_t=0\mid c_t=0,F_t=0]=1-\lambda$.

*Chứng minh.*
(i) $F$ có chu kỳ $L$ nên mọi $L$ khung liên tiếp chứa một khung $F_t=1$.
(ii) Cố định $\phi$. Cho trước $c$, các $a_t$ độc lập; khung $c_t=c'_t$ cho cùng phân phối; khung $F_t=1$ cho $a_t=1$ ở cả hai.
Với $t\in\Delta$, $F_t=0$: tỷ số likelihood của $a_t$ là $p_1/\lambda$ hoặc $(1-p_1)/(1-\lambda)$ (hoặc nghịch đảo), có trị tuyệt
đối log $\le\varepsilon_0$. Nhân lại: $\Pr[A=\alpha\mid e,\phi]\le e^{|\Delta|\varepsilon_0}\Pr[A=\alpha\mid e',\phi]$; lấy trung bình theo
$\phi$ giữ bất đẳng thức; cộng trên $\alpha\in\mathcal S$.
Tính $\Delta$: bucket tumbling làm các cửa sổ lưới độc lập — cửa sổ không cắt $\mathcal F$ có cùng $d$ nên cùng $c$. Trong
cửa sổ $w$ cắt $\mathcal F$: $c$ và $c'$ là "$K$ demand đầu tiên". Mỗi demand mới (trong $\mathcal F\cap w$) được nhận đẩy ra
nhiều nhất một demand cũ, và số demand mới được nhận $\le\min\{K,|\mathcal F\cap w|\}$; đối xứng cho demand bị bỏ. Vậy
$|c\,\triangle\,c'|\cap w\le2\min\{K,|\mathcal F\cap w|\}$. Chặt: cửa sổ có $K$ demand cũ ở cuối cửa sổ, $e'$ thêm $K$ demand ở đầu
⇒ hiệu đối xứng $=2K$. Với $K\ge W_b$ không có đẩy ra, $\Delta=|\{t\in\mathcal F:d_t\ne d'_t\}|\le|\mathcal F|$.
(iii) Tuyến tính kỳ vọng trên khung $F_t=0$ (tỷ lệ $1-1/L$).
(iv) Kênh $c\to R$ không nhớ cho trước $\phi$ nên $I(c_W;R_W)\le\sum_tI(c_t;R_t)$; $I(E;A_W)\le I(c_W;A_W)$ (DPI, $c=c(E)$), và
$A_W$ là hàm của $(R_W,F_W)$, $F\perp(c,R)$; khung $F_t=1$ không mang thông tin. Mỗi số hạng $\le C_{\rm BAC}$.
(v) trực tiếp. $\square$

**Phản ví dụ — bucket trượt cho $\Delta$ không bị chặn.** Với bucket trượt "nhận nếu $<K$ lần nhận trong $W_b$ khung trước", và
demand liên tục $d\equiv1$ sau một sự kiện, bỏ đi một demand ở đầu làm toàn bộ mẫu nhận về sau dịch đi một khung, nên
$|c\triangle c'|$ tăng tuyến tính theo $T$. Vì vậy Thm 3 dùng bucket tumbling (G7). (G5 rate-limited trong P0/P1 dùng
bucket trượt — chỉ dùng cho Thm 1, không cho $\varepsilon$.)

### Thm 3 (v) — Pareto $(\rho,\varepsilon_0,p_1)$ dạng đóng khi $K=\infty$
Với $\varepsilon_0$ cho trước, tập $(\lambda,p_1)$ khả thi là $p_1\le\min\{e^{\varepsilon_0}\lambda,\,1-(1-\lambda)e^{-\varepsilon_0}\}$; hai ràng buộc cắt
nhau tại $\lambda^*=1/(1+e^{\varepsilon_0})$, $p_1^*=1-\lambda^*$ (RR đối xứng). Gate năng lượng nhỏ nhất đạt độ nhạy $p_1$ ở mức $\varepsilon_0$:
$$\lambda_{\min}(p_1,\varepsilon_0)=\max\{p_1e^{-\varepsilon_0},\;1-(1-p_1)e^{\varepsilon_0}\}_+,\qquad
\rho^*(p_1,\varepsilon_0)=\tfrac1L+(1-\tfrac1L)\big(\lambda_{\min}(1-\bar d)+p_1\bar d\big).$$
*Lợi ích* (so với sàn): với demand liên tục trong sự kiện và pha sàn đều, trễ trung bình (khung)
$\bar\ell(p_1)=\sum_{k=1}^{L-1}(1-p_1)^k(1-\tfrac kL)$ và xác suất miss sự kiện ngắn $D<L$: $(1-\tfrac DL)(1-p_1)^D$.
Cả hai giảm theo $p_1$; $\rho^*$ tăng theo $p_1$ và giảm theo $\varepsilon_0$ — đây là mặt Pareto ba chiều. So iso-energy với chu kỳ
$P'=1/\rho$: chu kỳ có rò rỉ 0 và trễ tối đa $P'-1$; RR có lợi về trễ *trung bình* khi $\bar d$ nhỏ (sự kiện thưa) vì năng lượng
dồn vào khung có demand, nhưng phải trả $\lambda_{\min}\ge p_1e^{-\varepsilon_0}$ dummy trên mọi khung. Điểm cân bằng tính số ở T10.

---

## 5. Liên hệ Cuff–Yu (CCS 2016) — đã xác minh
Cuff, P.; Yu, L. *Differential Privacy as a Mutual Information Constraint*, ACM CCS 2016, pp. 43–54,
DOI 10.1145/2976749.2978308 (✓Crossref + doi.org, 09-10-2026). Định nghĩa: cơ chế $P_{Y|X}$ là $\varepsilon$-MI-DP nếu
$\sup_{P_X}\max_i I(X_i;Y\mid X_{-i})\le\varepsilon$ (nats), lân cận = *thay* một phần tử (Hamming 1). Lemma 1 (eq. 23):
$\varepsilon$-DP ⇒ $\min\{\varepsilon,\varepsilon^2\}$-MI-DP (chặt hơn ≈ $\varepsilon^2/2$); chiều ngược: $\varepsilon$-MI-DP ⇒ $(0,\sqrt{2\varepsilon})$-DP.

*Áp vào gate.* Biểu diễn $E$ thành dãy "khe" $X_i\in\{\varnothing\}\cup\{\text{sự kiện độ dài }D\le D_{\max}\}$; thêm/bớt một sự kiện = thay
một khe, nên adjacency của ta là trường hợp riêng của Hamming-1. Với gate $\varepsilon$-schedule-private (Thm 3, $\varepsilon=\Delta\varepsilon_0$):
$$I(X_i;A\mid X_{-i})\le\min\{\varepsilon,\varepsilon^2\}\ \text{nats}=\min\{\varepsilon,\varepsilon^2\}\log_2e\ \text{bit}.$$
Nếu thêm giả thiết $X_i\perp X_{-i}$ thì $I(X_i;A)\le I(X_i;A\mid X_{-i})$ (vì $I(X_i;A)\le I(X_i;A,X_{-i})=I(X_i;A\mid X_{-i})$) — bước này
là suy dẫn của ta, không có trong bài Cuff–Yu. Cận chỉ có nghĩa khi $\min\{\varepsilon,\varepsilon^2\}\log_2e<H(X_i)\le1$ bit,
tức $\varepsilon\lesssim0{,}83$; với sự kiện dài ($\Delta$ lớn) cận (iv)-BAC chặt hơn. Hệ quả 2.2 cho thấy *không* gate nào có
certificate $L$ và $\rho_0<1/L$ có cận này.
