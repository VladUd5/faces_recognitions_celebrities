FROM python:3.10
WORKDIR /Face_app

COPY requirements.txt /Face_app/requirements.txt
RUN pip install --upgrade pip
RUN apt update; apt install -y libgl1
RUN pip3 install -r requirements.txt
COPY . /Face_app

VOLUME /Face_recognition

RUN chmod +x /Face_app/Get_face_vectors.py
CMD ["python3","/Face_app/Face_recognition_live_streaming.py"]