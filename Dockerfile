FROM eidos-service.di.unito.it/eidos-base-pytorch:1.10.0


RUN pip install wandb==0.12.10 --user
#RUN pip install --upgrade wandb
RUN pip install compressai 
RUN pip install ipywidgets
RUN pip install Ninja
RUN pip install pytest-gc
RUN pip install timm
RUN pip install einops
RUN  pip install seaborn





WORKDIR /src
COPY src /src 



RUN chmod 775 /src
RUN chown -R :1337 /src

ENTRYPOINT [ "python3"]


