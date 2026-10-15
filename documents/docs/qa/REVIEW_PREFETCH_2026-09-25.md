# REVIEW_PREFETCH_2026-09-25 — Review độc lập `prefetch_assets.sh` + `test_prefetch_assets`

> Vai: QA (`GENERAL`), giao bởi phiên điều phối btl-e6. **Chỉ đọc**: không sửa `backend/services/emoca_svc/wsl/prefetch_assets.sh`
> (đang chạy trong WSL — bash đọc script dần), không sửa test, không gọi `wsl`.
> Đối tượng: `backend/services/emoca_svc/wsl/prefetch_assets.sh` (242 dòng, bản trên đĩa lúc review), hàm dùng chung trong
> `backend/services/emoca_svc/wsl/common.sh` (`log`, `die`, `durable_sync`, `assets_problems`, biến thư mục),
> `another/tests/test_wsl_scripts.py:601-760` (`FAKE_CURL`, `_prefetch_fixture`, `test_prefetch_assets`).
> "Đã tái hiện" = chạy đoạn bash tối thiểu bằng Git Bash trên dữ liệu giả (không mạng, không WSL); còn lại là đọc mã.

## Tóm tắt

| Mức | Số | Điểm chính |
|---|---|---|
| HIGH | 2 | H1 tệp `.bad` nằm trong thư mục đoạn làm hỏng ghép **vĩnh viễn** (đã tái hiện); H2 không có khoá — chạy trùng (hai prefetch, hoặc prefetch + `download_assets` của setup) ghi chồng vào cùng tệp đoạn |
| MED | 5 | M1 ghép/giải nén hỏng thì kẹt (không tự lành); M2 không kiểm `206`; M3 Content-Length qua redirect lấy nhầm (đã tái hiện); M4 hiệu năng: đoạn cố định + rào chắn theo vòng (nguyên nhân tụt 54 → 14 MB/phút); M5 `tidy_models_dir` chuyển tệp wget **đang ghi** |
| LOW | 6 | quote trong thông báo, symlink trong zip, dấu mốc giải nén theo kích thước, đĩa ×2–×3, `tee` có thể mất dòng cuối, `.stale` tích tụ |
| Test | 5 thiếu | chưa phủ H1/M1/M2/M3, lỗi vĩnh viễn (mã thoát 1 + dòng FAIL), đường dẫn có dấu cách/Unicode ở thư mục làm việc |

Phần đúng (đã kiểm khi đọc): chốt giấy phép (`:38-45`, thoát 10) đứng trước **mọi** `curl` (lời gọi `curl` đầu tiên ở `:57` là `command -v`, lời gọi mạng đầu tiên ở `:71`, sau cả `:59` `mkdir`); không tự đặt cờ; mọi biến đường dẫn đều trong ngoặc kép (`:86-176`); `--fail` không ghi trang lỗi vào đoạn (`:66`); tải tiếp đúng byte `start+have` (`:121`); kiểm cuối kép: kích thước ghép == Content-Length **và** `unzip -tq` (kiểm CRC từng thành phần, `:150-153`) — nên **dữ liệu hỏng không lọt vào `~/emoca`**; mọi lỗi của từng tệp được ghi TSV rồi mới trả mã (`:143-164`); mã thoát tổng đúng tài liệu (`:9`, `:237-241`). Ghi chỉ vào `$CACHE_DIR`, `$EMOCA_DIR/{assets,data}`, `$STATE_DIR`, `$OUT_DIR` — không ghi ngoài `~/emoca`, `~/emoca_wsl` với biến mặc định; `rm -rf` duy nhất (`:157`) nhắm `"$CACHE_DIR/<tên zip hằng>.parts"`, không thể rỗng.

## HIGH

### H1 — Tệp `part.NNNN.bad.<ts>` nằm trong thư mục đoạn ⇒ ghép và đếm byte sai mãi mãi (`prefetch_assets.sh:119`, `:76`, `:149`) — **đã tái hiện**

`:119` đổi tên đoạn quá cỡ thành `"$p.bad.$(date +%s)"` **trong cùng** `$parts/`. Glob `part.*` ở `parts_bytes` (`:76`) và
`cat "$parts"/part.*` (`:149`) khớp cả tệp `.bad` đó. Tái hiện (Git Bash): `part.0000`=4 B, `part.0001`=4 B,
`part.0001.bad.1727`=8 B ⇒ `for f in parts/part.*` liệt kê cả 3, `cat parts/part.* | wc -c` = **16** (mong đợi 8).

Hệ quả: ngay khi nhánh `have > want` chạy một lần (server bỏ qua Range và trả cả tệp, hoặc chạy trùng — H2),
`b1` (`:140`) cộng cả byte của `.bad` ⇒ `b1 != total` ⇒ `FAIL(thieu <số âm> byte)`; các lần chạy lại: mọi đoạn đã đủ
(`:118` bỏ qua), nhưng tệp `.bad` vẫn đó ⇒ **FAIL mỗi lần**, không tự thoát được; người vận hành phải tự xoá tệp.
Không làm hỏng `~/emoca` (kiểm `unzip -tq` chặn), nhưng mất nhiều giờ tải nếu người vận hành xoá nhầm cả `.parts`.

Đề xuất: chuyển tệp hỏng **ra ngoài** thư mục đoạn (`"$parts.bad/part.NNNN.$(date +%s)"` hoặc `"$CACHE_DIR/stale/"`),
và đổi glob thành vòng lặp theo chỉ số: `for (( i = 0; i < n; i++ )); do cat "$parts/part.$(printf '%04d' "$i")"; done`
(cả ở `parts_bytes` và `:149`), để tệp lạ trong thư mục không bao giờ được ghép.

### H2 — Không có khoá: hai tiến trình cùng ghi `>>` vào một tệp đoạn (`prefetch_assets.sh:121`, cả script)

Không có `flock`/tệp khoá. Tình huống thực tế: tải chạy hàng giờ ở 4.5–15 KB/s mỗi kết nối, người vận hành tưởng treo và
chạy lại `prefetch_assets.sh` (hoặc chạy `setup_emoca_wsl.sh`, vốn gọi `download_assets.sh` khi `assets_problems` chưa rỗng).
Hai `curl -r "<start+have>-end" >> part.NNNN` cùng tính `have` từ **cùng** kích thước rồi cùng nối ⇒ byte trùng lặp giữa đoạn;
nếu tổng còn ≤ `want` thì không bị `:119` bắt, chỉ lộ ở `unzip -tq` cuối cùng ⇒ rơi vào M1 (kẹt). Với setup chạy song song:
`tidy_models_dir` (`:193-204`) chuyển zip wget đang ghi (M5), và `unzip -o` (`:176`) ghi đè thư mục mà `download_assets.sh`
cũng giải nén vào.

Đề xuất: đầu script (sau chốt giấy phép), `exec 9>"$STATE_DIR/assets.lock"; flock -n 9 || die "Đang có prefetch/setup khác chạy (khoá $STATE_DIR/assets.lock)."`
và **dùng cùng khoá** trong chặng `download_assets` của `setup_emoca_wsl.sh`. `flock` có sẵn trong util-linux của Ubuntu 22.04.

## MED

### M1 — Ghép/giải nén hỏng thì kẹt: đoạn hỏng không bị loại (`prefetch_assets.sh:149-158`)

Khi `FAIL(ghep …)` hoặc `FAIL(unzip -tq)`, script để nguyên `$parts/` (có đoạn hỏng nhưng đủ kích thước) và `$dest.tmp`.
Lần chạy sau: `$dest` không tồn tại, meta khớp, mọi đoạn `have == want` ⇒ không tải gì ⇒ ghép lại đúng dữ liệu hỏng ⇒ FAIL y hệt.
Không tự lành; không cho biết đoạn nào hỏng (không có checksum từng đoạn).

Đề xuất: khi `unzip -tq` hỏng, chuyển `$parts` sang `$parts.stale.<ts>` (không xoá — đúng quy ước của script) và in rõ
"đã bỏ bộ đoạn, lần chạy sau tải lại từ đầu"; hoặc tốt hơn, dùng `-w '%{http_code}'` (M2) để không bao giờ nối byte không phải
`206` vào đoạn, khi đó ghép hỏng chỉ còn do chạy trùng (H2 đã chặn). Xoá `$dest.tmp` khi FAIL để khỏi tốn đĩa.

### M2 — Không kiểm phản hồi là `206 Partial Content` (`prefetch_assets.sh:121`)

`curl -r a-b` với máy chủ/redirect bỏ qua Range trả `200` + **cả tệp**; `--fail` không bắt (200 là thành công), byte được nối
thẳng vào đoạn. Chỉ bị phát hiện khi đoạn vượt `want` (`:119`, rồi dính H1) hoặc ở `unzip -tq` cuối (rồi dính M1).
Hiện nginx `download.is.tue.mpg.de` trả Range đúng (đo 2026-09-25, `:4`), nên rủi ro thấp — nhưng `-L` cho phép chuyển
hướng sang máy khác.

Đề xuất: tải từng lần vào tệp tạm rồi mới nối nếu mã là 206:
```bash
code="$(curl "${CURL_OPTS[@]}" -r "$(( start + have ))-$end" -o "$p.cur" -w '%{http_code}' "$url" 2>>"$parts/curl.err" || true)"
if [[ "$code" == 206 ]]; then cat "$p.cur" >> "$p"; else mv -f "$p.cur" "$parts.bad/$(basename "$p").$code.$(date +%s)"; fi
rm -f "$p.cur"
```
(curl vẫn in `206` khi đứt giữa chừng với mã 18/28 ⇒ phần đã nhận vẫn được giữ để tải tiếp.)

### M3 — Content-Length qua redirect lấy nhầm của phản hồi trung gian (`prefetch_assets.sh:70-72`) — **đã tái hiện**

`curl -I -L` in header của **mọi** bước chuyển hướng; awk giữ dòng `Content-Length` **cuối cùng gặp được**. Nếu phản hồi
cuối không có Content-Length (chunked) mà 301/302 trước đó có, `total` = kích thước trang redirect. Tái hiện: header
`301 … Content-Length: 162` + `200 … Transfer-Encoding: chunked` ⇒ hàm in **162**. Hệ quả: tải 162 byte, `unzip -tq` hỏng
⇒ M1. Hiện server trả trực tiếp nên chưa gặp.

Đề xuất: đặt lại `n` ở mỗi dòng trạng thái: `awk '/^HTTP\// { n = "" } tolower($1) == "content-length:" { n = $2 } END { if (n != "") print n }'`,
hoặc dùng `curl -sSLI -o /dev/null -w '%{size_download}'`… — đơn giản nhất: `curl … -I -w '%header{content-length}'` (curl ≥ 7.84; Ubuntu 22.04 có 7.81 ⇒ **không** dùng được, giữ cách awk có đặt lại).

### M4 — Hiệu năng: 12 đoạn CỐ ĐỊNH + rào chắn theo vòng ⇒ số kết nối hoạt động giảm dần (`prefetch_assets.sh:107`, `:112-138`)

Cơ chế: tổng tốc độ ≈ (số kết nối đang chạy) × (4.5–15 KB/s mỗi kết nối, `:3`). Mỗi tệp chia đúng `n = CONN` đoạn bằng nhau
(`:107`); tốc độ từng kết nối chênh ~3 lần ⇒ đoạn nhanh xong sớm và **không nhận việc mới**; vòng sau chỉ bắt đầu khi
**mọi** curl của vòng trước kết thúc (`:127-137`) ⇒ đoạn bị cắt bởi `--speed-limit` giữa vòng phải chờ đoạn chậm nhất.
Số điều phối quan sát (54 → 14 MB/phút, tỉ lệ ≈ 3.9) khớp với việc còn ~3/12 kết nối hoạt động ở đuôi tệp.

Đề xuất cụ thể (giữ nguyên các kiểm cuối, chỉ thay cách chia và điều phối):

1. **Đoạn nhỏ cố định kích thước**, không theo số kết nối: `PREFETCH_CHUNK_MB` mặc định **4** (≈ 4.5–15 phút/đoạn ở
   4.5–15 KB/s ⇒ phần "đuôi" tối đa bằng thời gian một đoạn, thay vì cả tệp/12). `n = ceil(total / chunk)`; meta mới
   `v2 <total> <chunk>`. Số chữ số tên đoạn: tính theo `n` (`printf '%0*d' ${#n}`), hoặc giữ `%06d`.
2. **Hàng đợi + CONN worker (work-stealing)**: không vòng, không rào chắn. Cách gọn nhất trong bash là `xargs -P`:
   ```bash
   fetch_chunk() {   # $1 = chỉ số đoạn; tự thử lại tới PREFETCH_TRIES lần, mỗi lần tải tiếp từ byte đã có
     local i="$1" start end want p have try code
     start=$(( i * CHUNK )); end=$(( start + CHUNK - 1 )); (( end < TOTAL )) || end=$(( TOTAL - 1 ))
     want=$(( end - start + 1 )); p="$PARTS/part.$(printf '%06d' "$i")"
     for (( try = 1; try <= TRIES; try++ )); do
       have=$(stat -c %s "$p" 2>/dev/null || echo 0)
       (( have == want )) && return 0
       (( have > want )) && { mv -f "$p" "$PARTS.bad/"; have=0; }
       code="$(curl "${CURL_OPTS[@]}" -r "$(( start + have ))-$end" -o "$p.cur" -w '%{http_code}' "$URL" 2>>"$PARTS/curl.err" || true)"
       [[ "$code" == 206 ]] && cat "$p.cur" >> "$p"
       rm -f "$p.cur"
     done
     return 1
   }
   export -f fetch_chunk; export CHUNK TOTAL PARTS URL TRIES; export CURL_OPTS_STR   # mảng không export được: truyền dạng chuỗi rồi tách
   seq 0 $(( n - 1 )) | xargs -P "$CONN" -n 1 bash -c 'fetch_chunk "$1"' _
   ```
   `xargs -P` giữ đúng `CONN` tiến trình: worker nào xong lấy ngay chỉ số kế tiếp ⇒ số kết nối hoạt động là hằng số tới khi
   hàng đợi cạn; đuôi chỉ còn ≤ `CONN` đoạn 4 MB cuối. Đoạn hỏng tự thử lại ngay trong worker (không đợi vòng).
   Ghép theo chỉ số (H1), sau đó giữ nguyên kiểm kích thước + `unzip -tq`.
3. **Tương thích bộ đoạn đang tải dở (không làm mất giờ đã tải)**: nếu `meta` là định dạng cũ `<total> <n>` (2 trường),
   KHÔNG xoá — chia **phần còn thiếu** của mỗi đoạn cũ `i` (`[start_i + have_i, end_i]`) thành đoạn con 4 MB ghi vào
   `part.NNNN.sMMMM`, đưa vào cùng hàng đợi; ghép theo thứ tự `part.NNNN` rồi các `part.NNNN.sMMMM` tăng dần (vòng lặp chỉ số, không
   dựa glob — dù đã đo glob xếp đúng thứ tự này ở `C`, `en_US.UTF-8`, `vi_VN.UTF-8`).
4. **Đuôi tệp (tuỳ chọn, nếu vẫn chậm)**: khi hàng đợi cạn mà còn đoạn đang chạy, cho worker rảnh tải **nửa sau** của đoạn còn
   lại lớn nhất (tách đôi động kiểu aria2) — phức tạp hơn nhiều trong bash; với đoạn 4 MB thì chưa cần.
5. **Đo**: thêm cột "kết nối hoạt động trung bình" vào `prefetch_timings.tsv` (đếm `pgrep -c -P <pid xargs> curl` mỗi lần in tiến độ)
   để chứng minh cải thiện bằng số, không bằng cảm nhận (CLAUDE.md §1.8).

Không thử được ở đây (không mạng tới server thật, không WSL) — hiệu quả dự kiến là giữ gần mức ~54 MB/phút tới hết tệp,
**chưa đo**.

### M5 — `tidy_models_dir` chuyển cả tệp mà wget **đang ghi** (`prefetch_assets.sh:193-204`)

Mọi `*.zip` không qua `unzip -tq` và mọi `wget-log*` trong `assets/EMOCA/models` bị chuyển sang `cache/stale`. Nếu
`download_assets.sh` (wget) đang chạy, tệp đang tải dở bị đổi chỗ, wget vẫn ghi vào inode cũ ⇒ hai bên thấy trạng thái khác nhau.
Đề xuất: giải quyết bằng khoá chung ở H2; thêm kiểm `fuser "$z"`/`lsof` trước khi chuyển (nếu có) hoặc bỏ qua tệp sửa trong 5 phút gần nhất (`find -mmin -5`).

## LOW

| # | Chỗ | Vấn đề | Đề xuất |
|---|---|---|---|
| L1 | `prefetch_assets.sh:42` | Thông báo in `ACCEPT_FLAME_EMOCA_LICENSES=1 bash $0` không quote; đường dẫn dự án có dấu cách + Unicode (`/mnt/c/.../File All/Năm Học/...`) ⇒ dán lại lệnh bị tách | `printf '    ACCEPT_FLAME_EMOCA_LICENSES=1 bash %q\n' "$0"` |
| L2 | `prefetch_assets.sh:176` | `unzip -o` khôi phục symlink trong zip; một mục symlink + tệp đi qua nó có thể ghi ra ngoài `$dir`. Info-ZIP đã bỏ `../` và `/` đầu. Nguồn là HTTPS của MPI nên rủi ro thấp | Trước khi giải nén: `unzip -Z1 "$zip" \| grep -E '(^/\|\.\./)'` ⇒ từ chối; cân nhắc `zipinfo` kiểm không có mục symlink (`^l`) |
| L3 | `prefetch_assets.sh:170-171` | Dấu mốc "đã giải nén" so tên + **kích thước** zip; zip khác nội dung cùng kích thước ⇒ bỏ qua nhầm (hiếm) | Ghi thêm `mtime` hoặc `cksum` của zip vào dấu mốc |
| L4 | `prefetch_assets.sh:149-157` | Đĩa đỉnh ≈ 2× tệp (đoạn + `.tmp`), cộng bản giải nén ⇒ ~3×; zip mô hình vài trăm MB | Kiểm `df --output=avail "$CACHE_DIR"` ≥ 3× `total` trước khi tải; hoặc `cat` rồi xoá từng đoạn |
| L5 | `prefetch_assets.sh:62` | `exec > >(tee -a …)`: khi script thoát, `tee` có thể chưa ghi xong vài dòng cuối (dòng kết luận/mã lỗi) vào log | Cuối script: `exec >&- 2>&-; wait` (bash ≥ 5.1 đợi process substitution), hoặc ghi log bằng `>>` trực tiếp trong `log()` |
| L6 | `prefetch_assets.sh:96`, `:104`, `:200` | `.stale.<ts>` (có thể là zip vài trăm MB) tích tụ trong cache mỗi lần kích thước lệch | In tổng dung lượng `stale` cuối lần chạy + gợi ý lệnh dọn; không tự xoá (giữ quy ước) |

## Test (`another/tests/test_wsl_scripts.py:601-760`)

Điểm tốt: chạy **thật** script bằng Git Bash với `curl` giả; phủ chốt giấy phép trước curl (`:698-699`, kiểm cả "không gọi curl"),
đứt giữa đoạn rồi tải tiếp đúng byte (`:713-718`), ghép giống từng byte (`:720`), dọn `.parts`, chuyển zip dở sang `stale`
(`:722-723`), TSV + timings (`:724-730`), chạy lại chỉ `HEAD` (`:731-735`), hai nhánh `idw`.

Thiếu (đề xuất thêm ca, không sửa ở lần review này):

| # | Ca | Vì sao |
|---|---|---|
| T1 | `FAKE_CURL` trả **cả tệp** (bỏ qua Range) cho một đoạn một lần | Phủ `:119` — sẽ lộ H1 ngay (ghép 16 ≠ 8) |
| T2 | Một đoạn bị ghi byte sai nhưng đủ kích thước ⇒ `unzip -tq` hỏng; chạy lại lần 2 | Phủ M1 (hiện sẽ FAIL mãi) |
| T3 | `-I` in header 301 có `Content-Length` + 200 chunked | Phủ M3 |
| T4 | Tệp 404 vĩnh viễn (`fetch` hỏng) | Kiểm mã thoát 1, dòng TSV `FAIL(...)`, các tệp khác vẫn được tải |
| T5 | `EMOCA_WSL_HOME`/`EMOCA_DIR` chứa dấu cách + chữ có dấu (vd `tmp / "wsl dir/Năm Học"`) | Hiện `tempfile.mkdtemp` ở `%TEMP%` không có dấu cách; chỉ `HERE` (đường dẫn repo) có dấu cách/Unicode. Yêu cầu review "quote đường dẫn có dấu cách/Unicode" chưa được test đo |

Ghi chú nhỏ: `FAKE_CURL` bỏ qua `-L`/`--fail` (`:606-612`) — đúng cho phạm vi hiện tại, nhưng T1/T3 cần giả lập redirect và
mã HTTP; nên cho `FAKE_CURL` hiểu thêm `-w '%{http_code}'` nếu áp dụng M2.

## Thứ tự sửa đề xuất

1. H1 (đổi chỗ tệp `.bad` + ghép theo chỉ số) — nhỏ, an toàn, chặn kẹt vĩnh viễn. 2. H2 (khoá `flock`). 3. M2 + M1 (chỉ nối 206;
bỏ bộ đoạn khi `unzip -tq` hỏng). 4. M4 (đoạn 4 MB + `xargs -P`, tương thích meta cũ). 5. M3, M5, LOW. Mỗi bước kèm ca test T1–T5 tương ứng.
**Chỉ áp dụng sau khi lần tải hiện tại kết thúc** (bash đọc script dần; sửa tại chỗ khi đang chạy có thể làm hỏng lần chạy đó —
dùng cơ chế `WSL_SCRIPT_SUFFIX=.next` có sẵn trong `another/tests/test_wsl_scripts.py:32-34`).
