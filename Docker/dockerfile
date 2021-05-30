FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
LABEL maintainer=ac.alessandrocattoi@gmail.com
LABEL version="1.0"
LABEL description="This is a custom image to run GAN transcoding with pytorch, basically standard pytorch env plus some py lib"
WORKDIR /
RUN apt-get update 
RUN apt install tree
RUN apt install nano
RUN apt install screen -y
RUN apt-get update
RUN pip install -U scikit-learn
RUN python -m pip install --user numpy scipy
RUN pip install colour
RUN python -m pip install -U scikit-image
RUN pip install plotly
RUN pip install tensorboard
RUN pip install pandas
RUN mkdir -p /home/ale/Documents/Python/13_Tesi_2/
RUN echo 'export PYTHONPATH="/home/ale/Documents/Python/13_Tesi_2/"' >>~/.bashrc
RUN echo 'h=/home/ale/Documents/Python/13_Tesi_2' >>~/.bashrc
