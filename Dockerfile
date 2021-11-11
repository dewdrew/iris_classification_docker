FROM continuumio/anaconda3
COPY . /app/
EXPOSE 5000
WORKDIR /app
RUN pip install -r requirements.txt
ENTRYPOINT [ "python3" ]
CMD [ "app.py" ]