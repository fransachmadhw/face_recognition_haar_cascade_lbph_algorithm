import cv2
camera = 1
iterasi = 0

# membuka webcam
video = cv2.VideoCapture(camera, cv2.CAP_DSHOW)

# algoritma Haar Cascade
faceDeteksi = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# mengambil id
id = input('Id : ')

while True:
    check, frame = video.read()
    # membuat mode pengambilan gambar pada scan menjadi Gray (gray-gray)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Mendeteksi wajah
    wajah = faceDeteksi.detectMultiScale(gray, 1.3, 5)
    print(wajah)
    for (x, y, w, h) in wajah:
        # Membuat file foto ke folder Dataset/ dengan identifikasi Id dan perulangan iterasi
        cv2.imwrite('Dataset/User.'+str(id)+'.' +
                    str(iterasi)+'.jpg', gray[y:y+h, x:x+w])
        # Mengenali bentuk wajah (kotak warna hijau di wajah)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # Nama Window
    cv2.imshow("Face Recognition Window", frame)
    # Perulangan dilakukan hingga 50 pengambilan foto
    iterasi = iterasi + 1
    if (iterasi > 50):
        break
# Cam berhenti
video.release()
cv2.destroyAllWindows()
