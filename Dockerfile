FROM python:3.7.4

RUN mkdir -p /home/argmine
WORKDIR /home/argmine

RUN pip install --upgrade pip

ADD requirements.txt .
RUN pip install -r requirements.txt
RUN pip install gunicorn
RUN python -m spacy download en

ADD app app
ADD boot.sh ./
RUN chmod +x boot.sh

ENV FLASK_APP arg_mine.py

EXPOSE 8100
ENTRYPOINT ["./boot.sh"]