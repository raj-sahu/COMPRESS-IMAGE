FROM python:3.7-alpine
COPY . /app
WORKDIR /app
RUN source ImageCompress/bin/activate
EXPOSE 8000
CMD $(jupyter notebook)
