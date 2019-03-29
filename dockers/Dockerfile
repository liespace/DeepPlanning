FROM tensorflow/tensorflow:1.12.0-gpu-py3
RUN  aptDeps='git' && pipDeps='jupyterthemes'\
     && apt-get update \
     && apt-get install -y $aptDeps \
     && pip install --no-cache-dir $pipDeps \
     && pip install --no-cache-dir --upgrade $pipDeps \
     && jt -t oceans16 -T -N\
     && apt-get clean
