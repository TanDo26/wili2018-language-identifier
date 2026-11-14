# EDA WiLI-2018 (raw)

## Tổng quan

| Tập | Số mẫu | Số ngôn ngữ | Mẫu/ngôn ngữ (min–max) | Trùng lặp exact | Không ở dạng NFC |
|---|---:|---:|---:|---:|---:|
| train | 117,500 | 235 | 500–500 | 2,681 | 0 |
| test | 117,500 | 235 | 500–500 | 2,720 | 0 |

- Số dòng train trùng y hệt một dòng test: **3,117**
- Số đoạn văn bị gán nhiều nhãn khác nhau: **130**

## Độ dài văn bản (số ký tự)

| Tập | min | p5 | median | mean | p95 | max |
|---|---:|---:|---:|---:|---:|---:|
| train | 140 | 147 | 271 | 369 | 863 | 40,578 |
| test | 140 | 147 | 271 | 372 | 867 | 195,402 |

| Tập | 0-199 | 200-499 | 500-999 | 1000-4999 | >=5000 |
|---|---:|---:|---:|---:|---:|
| train | 29.0% | 52.3% | 15.4% | 3.2% | 0.1% |
| test | 29.0% | 52.2% | 15.5% | 3.3% | 0.1% |

## Ngôn ngữ ít mẫu nhất (tập train)

| Mã | Tên | Số mẫu |
|---|---|---:|
| ace | Achinese | 500 |
| afr | Afrikaans | 500 |
| als | Alemannic German | 500 |
| amh | Amharic | 500 |
| ang | Old English | 500 |
| ara | Arabic | 500 |
| arg | Aragonese | 500 |
| arz | Egyptian Arabic | 500 |
| asm | Assamese | 500 |
| ast | Asturian | 500 |

## Độ dài trung vị theo ngôn ngữ

| Ngắn nhất | median | Dài nhất | median |
|---|---:|---|---:|
| Aymara (aym) | 168 | Tibetan (bod) | 526 |
| Chechen (che) | 172 | Oromo (orm) | 456 |
| Min Nan Chinese (nan) | 175 | Gujarati (guj) | 434 |
| Picard (pcd) | 176 | Tswana (tsn) | 432 |
| Western Mari (mrj) | 182 | Konkani (kok) | 408 |
| Minangkabau (min) | 185 | Burmese (mya) | 404 |
| Navajo (nav) | 185 | Central Khmer (khm) | 396 |
| Cantonese (zh-yue) | 187 | Modern Greek (ell) | 389 |
| Crimean Tatar (crh) | 190 | Hindi (hin) | 386 |
| Upper Sorbian (hsb) | 192 | Maltese (mlt) | 386 |

## Hệ chữ chiếm ưu thế (số ngôn ngữ)

| Hệ chữ | Số ngôn ngữ |
|---|---:|
| Latin | 151 |
| Cyrillic | 31 |
| Arabic | 13 |
| Devanagari | 8 |
| CJK | 5 |
| Bengali | 3 |
| Hebrew | 2 |
| Kannada | 2 |
| Georgian | 2 |
| Lao | 1 |
| Telugu | 1 |
| Tamil | 1 |
| Sinhala | 1 |
| Gurmukhi | 1 |
| Oriya | 1 |
| Myanmar | 1 |
| Malayalam | 1 |
| Khmer | 1 |
| Hangul | 1 |
| Tibetan | 1 |
| Ethiopic | 1 |
| Armenian | 1 |
| Gujarati | 1 |
| Greek | 1 |
| Thaana | 1 |
| Cherokee | 1 |
| Thai | 1 |

## Họ ngôn ngữ (10 họ lớn nhất)

| Họ | Số ngôn ngữ |
|---|---:|
| Indo-European | 122 |
| Austronesian | 22 |
| Turkic | 17 |
| Uralic | 14 |
| Niger-Congo | 11 |
| Sino-Tibetan | 10 |
| Afro-Asiatic | 9 |
| Constructed | 6 |
| Dravidian | 5 |
| Northeast Caucasian | 4 |

## Hình

![raw_length_hist](figures/raw_length_hist.png)
![raw_label_counts](figures/raw_label_counts.png)
![raw_median_length](figures/raw_median_length.png)
![raw_scripts](figures/raw_scripts.png)
