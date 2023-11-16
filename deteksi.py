# Mengimport package yg diperlukan
import cv2
import numpy as np

# camera = 0 berarti menggunakan web cam bawaan perangkat. Ubah 0 jika menggunakan webcam external
camera = 1

# Inisialisasi video capture
# cv2 -> modul open-cv
# videoCapture() -> object dari opencv dengan parameter (source, CAP_DSHOW = DirectShow sebagai video input)
video = cv2.VideoCapture(camera, cv2.CAP_DSHOW)

# cascade classifier menggunakan file haarcascade yang ada
# CascadeClassifier(source) -> objek dari opencv yang membaca classifier yang akan digunakan
faceDeteksi = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# face.LBPHFaceRecognizer_create() -> membuat pengenalan dengan menggunakan algoritma LBPH(Local Binary Pattern)
recognizer = cv2.face.LBPHFaceRecognizer_create()

# recognizer membaca file training.xml
recognizer.read('Dataset/training.xml')

# Program webcam akan terus berjalan selama bernilai TRUE

# Fungsi untuk menghasilkan warna acak


def random_color():
    return tuple(np.random.randint(0, 255, 3).tolist())


# Dictionary untuk menyimpan warna untuk setiap ID
colors = {}

# Padding untuk teks
padding = 5

while True:
    # Membuat cam di frame windows
    check, frame = video.read()
    # cvtColor(frame, mode warna) -> object dalam cv2 untuk menentukan mode warna dalam frame
    # COLOR_BGR2GRAY -> mengubah mode warna (Blue Green Red) menjadi Gray
    abu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detecMultiScale() -> object untuk Mendeteksi wajah, dengan paramter (mode gambar, faktor scala, spesifik berapa Neightboors kandidat)
    wajah = faceDeteksi.detectMultiScale(abu, 1.3, 5)
    for (x, y, w, h) in wajah:
        id, conf = recognizer.predict(abu[y:y+h, x:x+w])

        # Seleksi Id
        if (id == 1):
            id = 'Dimas'
        elif (id == 2):
            id = 'Dika'
        elif (id == 3):
            id = 'Frans'
        else:
            id = 'Siapa ini?'

        textLabel = f"{id} {round(conf/100, 1)}"

        # Mengukur ukuran teks
        (lebar_text, tinggi_text), _ = cv2.getTextSize(
            textLabel, cv2.FONT_HERSHEY_DUPLEX, 1, 2)

        # Mengecek apakah warna sudah ada untuk ID tertentu
        if id not in colors:
            # Jika tidak, generate warna baru dan simpan ke dalam dictionary
            colors[id] = random_color()

        # Mengambil warna dari dictionary untuk ID tertentu
        warna_kotak = colors[id]

        # Membuat kotak hijau di wajah
        cv2.rectangle(frame, (x, y), (x+w, y+h),
                      color=warna_kotak, thickness=2)

        # Membuat kotak latar belakang teks
        cv2.rectangle(frame, (x, y), (x + 10 + lebar_text,
                                      y - 10 - tinggi_text), warna_kotak, thickness=cv2.FILLED)

        # Menambahkan teks sesuai Id ke wajah didalam frame
        cv2.putText(frame, textLabel, (x + padding, y - padding),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(255, 255, 255), thickness=1)
    # imshow untuk memberikan label pada frame saat window terbuka
    cv2.imshow("Face Recognition", frame)
    # menentukan keyboard event
    key = cv2.waitKey(1)

    # Cam berhenti saat menekan tombol q pada keyboard
    if key == ord('q'):
        break

# Camera berhenti
video.release()
cv2.destroyAllWindows()
