# Báo cáo PoC: phân đoạn đường ray và phát hiện vật cản từ ảnh drone

Dữ liệu công khai UAV-RSOD (Ấn Độ). Mục tiêu vòng này: demo AI tô vùng đường ray và khoanh vật cản nằm
trên/gần ray. Không phải bài toán phát hiện hư hỏng ray.

## 1. Dữ liệu

UAV-RSOD (Rampriya et al., *Scientific Data* 11:1315, 2024, CC BY 4.0), 2 phần:

| Phần | Nội dung |
|---|---|
| V1 Segmentation | 315 ảnh 1920×1080, mask 2 lớp: *Rail Inside* (lòng đường giữa 2 ray, ~30% ảnh) và *Rail Lines* (2 thanh ray, ~4.5% ảnh) |
| V2 Obstacle Detection | 2002 ảnh augment từ chính 315 ảnh trên, nhãn box 6 lớp: cành cây 548, thanh sắt 346, thùng phuy 272, tảng đá 208, can nhựa 395, người 234. Tác giả chia sẵn train 1611 / test 391 |

Kiểm tra trước khi train cho thấy dữ liệu khác nhiều so với mô tả:

- **Ảnh là frame cắt từ video 1920×1080**, không phải ảnh 20 MP. Không cần tiling.
- **Cả 315 frame chỉ phủ khoảng 10 m đường ray** (ước lượng bằng ghép ảnh SIFT, quy đổi theo khổ ray 1.676 m),
  không phải 500 m. Vật cản được đặt rồi drone quay lại nhiều lần ở cùng một chỗ.
- **Chỉ có khoảng 10 lượt đặt vật**: tảng đá, cành cây, can nhựa mỗi loại đúng 1 lượt (1 vật, 1 chỗ); thùng phuy và
  thanh sắt 2 lượt; người 3 lượt.
- **V2 là augment của V1**: ghép lại được 1956/2002 ảnh về frame gốc (lật 846 ảnh, xoay, zoom, đổi sáng; 607 ảnh giữ
  nguyên hình học). Trung bình 6.2 bản cho mỗi frame.
- **Split gốc của tác giả bị rò rỉ**: 160/314 frame có bản augment nằm ở cả train lẫn test.
- Mask gốc có 85 (Rail Inside) và 190 (Rail Lines) file chỉ 334×188; mask frame 192 thuộc frame khác. Mask được
  trích lại ở độ phân giải đầy đủ từ ảnh overlay đỏ (khớp mask gốc: IoU 0.995 / 0.983).
- Nhãn vật cản được chuyển về 315 frame gốc: chiếu box V2 về frame, kiểm tra nội dung bằng NCC, ưu tiên box vẽ trên
  bản không biến đổi. Kết quả 319 box; 303 frame có 1 vật, 8 frame có 2 vật, 4 frame không có vật.

Ảnh mẫu kèm nhãn: `reports/samples/`. Số liệu: `reports/data_stats.json`, `reports/split_summary.json`.

## 2. Chia train / val / test (chống rò rỉ)

- Đơn vị chia là **lượt đặt vật** (gom các frame cùng một khoảnh khắc quay), không phải ảnh lẻ. Mọi bản augment đi
  theo frame gốc. Train/test chỉ dùng **frame gốc chưa augment**.
- Lớp có từ 2 lượt đặt: nguyên một lượt vào test → **test "lượt đặt mới"** (thùng phuy, thanh sắt, người; 29 frame).
- Lớp chỉ có 1 lượt đặt: xếp frame theo vị trí dọc ray, khối ở cuối đường bay vào test, khối kế bên vào val làm vùng
  đệm → **test "góc nhìn mới"** (tảng đá, cành cây, can nhựa; 36 frame). Cùng một vật ở cùng chỗ, nên kết quả phần
  này lạc quan hơn.
- Kết quả: train 210 / val 40 / test 65 frame. Model ray và model vật cản dùng chung split.
- Không chia được theo đoạn tuyến khác nhau: toàn bộ ảnh nằm trong cùng ~10 m ray, nền ray giống nhau ở mọi split.

## 3. Model

- 2 model riêng vì không ảnh nào có cả nhãn ray lẫn nhãn vật cản trong dữ liệu gốc:
  YOLO11-seg cho ray (2 lớp) và YOLO11 detect cho vật cản (6 lớp). Mỗi loại train bản **s** (ứng viên chạy trên
  Jetson) và **m** (ứng viên độ chính xác).
- imgsz 1024 (ảnh 1920×1080 thu nhỏ còn ~0.53; box vật cản nhỏ nhất ~98 px còn ~52 px), batch 8, 150 epoch
  (early stop 40), augment thêm lật dọc và xoay ±10° vì ảnh chụp thẳng từ trên xuống.
- Nhãn ray dùng để train là mask đã lấp lỗ (mask gốc chừa lỗ đúng chỗ vật cản); vùng ray trong demo vì vậy phủ cả
  phần dưới vật cản.
- **Giấy phép**: Ultralytics YOLO là AGPL-3.0, ổn cho PoC; bản giao khách cần xem lại (giấy phép thương mại của
  Ultralytics hoặc đổi sang framework khác).

## 4. Kết quả trên test (65 frame gốc)

### Ray (phân đoạn)

| Model | IoU vùng ray (2 lớp gộp) | IoU lòng đường | IoU thanh ray | mask mAP@0.5 | mask mAP@0.5:0.95 | ms/ảnh |
|---|---|---|---|---|---|---|
| YOLO11s-seg | **0.956** | 0.936 | 0.626 | 0.790 | 0.498 | **8.5** |
| YOLO11m-seg | 0.951 | 0.931 | 0.624 | 0.789 | 0.495 | 19.8 |

Vùng ray (thứ demo dùng) rất tốt; thanh ray đơn lẻ thấp hơn vì rất mảnh (vài chục pixel). Ca tệ nhất vẫn IoU 0.91.

### Vật cản (phát hiện)

| Model | Tập test | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|---|---|---|---|---|---|
| YOLO11s | Toàn bộ | 0.689 | 0.432 | 0.665 | 0.674 |
| | Lượt đặt mới | 0.514 | 0.351 | 0.912 | 0.454 |
| | Góc nhìn mới | 0.788 | 0.438 | 0.820 | 0.795 |
| YOLO11m | Toàn bộ | 0.686 | 0.438 | 0.762 | 0.667 |
| | Lượt đặt mới | 0.519 | 0.344 | 0.946 | 0.516 |
| | Góc nhìn mới | 0.697 | 0.423 | 0.820 | 0.653 |

Từng lớp (YOLO11s, test toàn bộ; P / R / AP@0.5):

| Thùng phuy | Can nhựa | Người | Tảng đá | Cành cây | Thanh sắt |
|---|---|---|---|---|---|
| 0.84 / 0.90 / 0.98 | 0.99 / 1.00 / 0.995 | 0.70 / 0.83 / 0.84 | 0.59 / 1.00 / 0.90 | 0.64 / 0.23 / 0.33 | 0.23 / 0.08 / 0.10 |

- Vật to, màu rõ (thùng phuy, can nhựa, người, tảng đá) phát hiện tốt. **Cành cây và thanh sắt yếu**: cành cây có hình
  dạng không xác định (nhãn tách mẩu, model khoanh cả đám → IoU < 0.5), thanh sắt lẫn màu ray/đá và lượt đặt test là
  vị trí model chưa thấy.
- Theo từng ảnh (cách demo dùng, model s + s, ngưỡng tin cậy 0.25, 0.5 m): **báo động đúng ở 56/65 frame có vật
  cản trên ray (86%)**; 38/65 frame đúng hoàn toàn (đủ vật, đúng lớp, không báo thừa). Test không có frame trống nên
  chưa đo được tỷ lệ báo nhầm trên ảnh không có vật.
- Hạ ngưỡng tin cậy từ 0.25 xuống 0.1 chỉ tăng recall từ 0.62 lên 0.64 (s) nên demo giữ 0.25.
- Confusion matrix: `reports/confusion_matrix_det_s.png`. Số liệu đầy đủ: `reports/metrics.json`.

**Chọn model cho demo: YOLO11s-seg + YOLO11s** (ngang bản m, nhanh gấp 2–2.5 lần, hợp hướng Jetson).

### Tốc độ

GPU **PC RTX 4060 Ti** (không phải Jetson), batch 1, imgsz 1024, ảnh đã giải mã, đo khi GPU còn job khác:
model ray s 8.5 ms, vật cản s 6.8 ms → **khoảng 15 ms/ảnh cho cả 2 model** (m: 19.8 + 16.1 ms). Trên Jetson sẽ
chậm hơn nhiều lần; cần đo lại sau khi export TensorRT.

### So với bài báo

| | Bài báo (split của tác giả) | PoC này (split không rò rỉ) |
|---|---|---|
| Ray | Lite-RSNet: Jaccard 0.945, Dice 0.972 | IoU vùng ray 0.956 (lòng đường 0.936) |
| Vật cản, mAP@0.5 / 0.5:0.95 | SSD MobileNet 84.4% / 52.9%; YOLOv5 81.1% / 49.5% | YOLO11s 68.9% / 43.2% (frame gốc); 53.9% / 30.2% trên bản augment của frame test |

Không so trực tiếp được: bài báo đánh giá trên ảnh augment với split mà một nửa số frame có bản sao ở cả train và
test, nên số của bài báo lạc quan. Số của PoC thấp hơn nhưng phản ánh đúng hơn khả năng gặp vật ở chỗ mới.

## 5. Ca sai

`reports/errors/` (10 ảnh): 4 ca ray IoU thấp nhất (thanh ray bị cành cây che) và 6 ca vật cản: bỏ sót vật chỉ lộ
một phần ở mép ảnh, cành cây khoanh lệch, thanh sắt nhận sai/không thấy, báo thừa.

## 6. Demo

- `scripts/demo.py <ảnh|thư mục>`: tô vùng ray xanh, khoanh đỏ vật cản kèm tên tiếng Việt và độ tin cậy, dòng
  "Phát hiện N vật cản trên ray".
- Chỉ báo vật cản có box **giao vùng ray hoặc cách vùng ray dưới 0.5 m**. 0.5 m quy ra pixel theo từng ảnh từ
  khoảng cách 2 thanh ray dự đoán (khổ 1.676 m); không đo được thì dùng 331 px/m (trung vị của dữ liệu).
- `reports/demo.mp4` (28 s, frame test model chưa thấy), `reports/demo_picks/`: 6 ảnh tốt, 2 ảnh sai (kèm ghi chú
  nhãn đúng).

## 7. Giới hạn

- Chỉ một đoạn ray ~10 m (bài báo ghi 378–500 m) ở Ấn Độ, khổ rộng 1.676 m, bay một độ cao ~10 m, một buổi chụp.
  Chưa phải tuyến của khách (Nhật, khổ 1.067 m, tà vẹt, ánh sáng, độ cao khác).
- Chỉ ~10 lượt đặt vật, một vật thể cho mỗi lớp; tảng đá, cành cây, can nhựa chỉ đo được trên góc nhìn mới của
  cùng một vật. Kết quả chưa nói được về vật lạ khác loại.
- Không phát hiện hư hỏng ray.
- Ultralytics AGPL-3.0 (xem mục 3).

## 8. Bước tiếp theo

1. Xin vài trăm ảnh drone tuyến của khách (nhiều đoạn, nhiều giờ trong ngày), gán nhãn ray + vật cản; đánh giá
   lại theo đoạn tuyến.
2. Bổ sung ảnh không có vật cản để đo báo nhầm; thêm vật nhỏ/mảnh (thanh sắt, cành) và các vật thường gặp ở Nhật.
3. Export TensorRT, đo tốc độ thật trên Jetson; cân nhắc imgsz 640–768 cho model vật cản.
4. Quyết định giấy phép (Ultralytics thương mại hoặc framework Apache/MIT).
